from __future__ import annotations

"""

Tool to add default values to stubs.

"""

import argparse
import ast
import contextlib
import importlib
import inspect
import io
import math
import subprocess
import sys
import textwrap
import types
import typing
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Type, Union, cast

import libcst
import tomli
import typeshed_client
from termcolor import colored

# Defaults with a repr longer than this number will not be added.
# This is an arbitrary value, but it's useful to have a cut-off point somewhere.
# There's no real use case for *very* long defaults,
# and they have the potential to cause severe performance issues
# for tools that try to parse or display Python source code
DEFAULT_LENGTH_LIMIT = 500


def default_is_too_long(default: libcst.BaseExpression) -> bool:
    default_as_module = libcst.Module(
        body=[libcst.SimpleStatementLine(body=[libcst.Expr(value=default)])]
    )
    repr_of_default = default_as_module.code.strip()
    return len(repr_of_default) > DEFAULT_LENGTH_LIMIT


def log(*objects: object) -> None:
    print(colored(" ".join(map(str, objects)), "yellow"))


def infer_value_of_node(node: libcst.BaseExpression) -> object:
    """Return NotImplemented if we can't infer the value."""
    if isinstance(node, (libcst.Integer, libcst.Float, libcst.SimpleString)):
        return node.evaluated_value
    elif isinstance(node, libcst.Name):
        if node.value == "True":
            return True
        elif node.value == "False":
            return False
        elif node.value == "None":
            return None
        else:
            return NotImplemented
    elif isinstance(node, libcst.UnaryOperation):
        if isinstance(node.operator, libcst.Minus):
            operand = infer_value_of_node(node.expression)
            if not isinstance(operand, (int, float)):
                return NotImplemented
            return -operand
        else:
            return NotImplemented
    elif isinstance(node, (libcst.List, libcst.Tuple, libcst.Set)):
        ret = [infer_value_of_node(element.value) for element in node.elements]
        if NotImplemented in ret:
            return NotImplemented
        elif isinstance(node, libcst.List):
            return ret
        elif isinstance(node, libcst.Tuple):
            return tuple(ret)
        else:
            return set(ret)
    elif isinstance(node, libcst.Dict):
        dict_ret = {}
        for element in node.elements:
            if isinstance(element, libcst.DictElement):
                key = infer_value_of_node(element.key)
                if key is NotImplemented:
                    return NotImplemented
                value = infer_value_of_node(element.value)
                if value is NotImplemented:
                    return NotImplemented
                dict_ret[key] = value
            else:
                return NotImplemented
        return dict_ret
    else:
        return NotImplemented


@dataclass
class ReplaceEllipsesUsingRuntime(libcst.CSTTransformer):
    sig: inspect.Signature
    stub_params: libcst.Parameters
    num_added: int = 0
    errors: List[Tuple[str, object, object]] = field(default_factory=list)

    def get_matching_runtime_parameter(
        self, node: libcst.Param
    ) -> inspect.Parameter | None:
        param_name = node.name.value

        # Scenario (1): the stub signature and the runtime signature have parameters with the same name
        # Assume identically named parameters are "the same" parameter;
        # return the runtime parameter with the same name
        try:
            return self.sig.parameters[param_name]
        except KeyError:
            pass

        # Scenario (2): the stub signature has a parameter with the same name as a parameter at runtime,
        # except that the parameter in the stub has `__` prepended to the beginning of the name.
        # This is used in stubs to indicate positional-only parameters on Python <3.8;
        # assume that these similarly named parameters are also "the same" parameter;
        # return the runtime parameter with the similar name
        if (
            node in self.stub_params.params
            and param_name.startswith("__")
            and not param_name.endswith("__")
        ):
            try:
                return self.sig.parameters[param_name[2:]]
            except KeyError:
                pass
        elif node not in self.stub_params.posonly_params:
            return None

        # Scenario (3): the runtime signature doesn't have any parameters
        # that have the same name as the stub parameter,
        # or that have the same name with `__` prepended.
        # Fall back to the nth parameter of the runtime signature
        # (where n is the index of the stub parameter in the stub signature),
        # iff all the following conditions are true:
        #
        # - The number of parameters at runtime == the number of parameters in the stub
        # - There are no variadic parameters (*args or **kwargs) at runtime or in the stub
        # - The parameter is marked as being pos-only in the stub,
        #   either through PEP 570 syntax or through a parameter name starting with `__`
        # - The parameter is also positional-only at runtime
        all_runtime_parameters = list(self.sig.parameters.values())
        variadic_parameter_kinds = {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }

        if any(
            param.kind in variadic_parameter_kinds for param in all_runtime_parameters
        ):
            return None

        if isinstance(self.stub_params.star_arg, libcst.Param) or isinstance(
            self.stub_params.star_kwarg, libcst.Param
        ):
            return None

        all_stub_params = list(
            chain(
                self.stub_params.posonly_params,
                self.stub_params.params,
                self.stub_params.kwonly_params,
            )
        )

        if len(all_stub_params) != len(all_runtime_parameters):
            return None

        runtime_param = all_runtime_parameters[all_stub_params.index(node)]
        if runtime_param.kind is inspect.Parameter.POSITIONAL_ONLY:
            return runtime_param
        return None

    def infer_value_for_default(
        self, node: libcst.Param
    ) -> libcst.BaseExpression | None:
        param = self.get_matching_runtime_parameter(node)
        if not isinstance(param, inspect.Parameter):
            return None
        if param.default is inspect.Parameter.empty:
            return None
        new_stub_default = self._infer_value_for_default(node, param.default)
        if new_stub_default is None or default_is_too_long(new_stub_default):
            return None
        return new_stub_default

    def _infer_value_for_default(
        self, node: libcst.Param | None, runtime_default: Any
    ) -> libcst.BaseExpression | None:
        if isinstance(runtime_default, (bool, type(None))):
            return libcst.Name(value=str(runtime_default))
        elif type(runtime_default) in {str, bytes}:
            return libcst.SimpleString(value=repr(runtime_default))
        elif type(runtime_default) is int:
            if (
                node
                and node.annotation
                and isinstance(node.annotation.annotation, libcst.Name)
                and node.annotation.annotation.value == "bool"
            ):
                # Skip cases where the type is annotated as bool but the default is an int.
                return None
            if runtime_default >= 0:
                return libcst.Integer(value=str(runtime_default))
            else:
                return libcst.UnaryOperation(
                    operator=libcst.Minus(),
                    expression=libcst.Integer(value=str(-runtime_default)),
                )
        elif type(runtime_default) is float:
            if not math.isfinite(runtime_default):
                # Edge cases that it's probably not worth handling
                return None
            # `-0.0 == +0.0`, but we want to keep the sign,
            # so use math.copysign() rather than a comparison with 0
            # to determine whether or not it's a negative float
            if math.copysign(1, runtime_default) < 0:
                return libcst.UnaryOperation(
                    operator=libcst.Minus(),
                    expression=libcst.Float(value=str(-runtime_default)),
                )
            else:
                return libcst.Float(value=str(runtime_default))
        elif type(runtime_default) in {tuple, list}:
            members = [
                self._infer_value_for_default(None, member)
                for member in runtime_default
            ]
            if None in members:
                return None
            # pyanalyze doesn't like us using lowercase type[] here on <3.9
            libcst_cls: Type[libcst.Tuple | libcst.List]
            libcst_cls = (
                libcst.Tuple if isinstance(runtime_default, tuple) else libcst.List
            )
            return libcst_cls(
                elements=[
                    libcst.Element(cast(libcst.BaseExpression, member))
                    for member in members
                ]
            )
        elif type(runtime_default) is set:
            if not runtime_default:
                # The empty set is a "call expression", not a literal;
                # we only want to add defaults where they can be expressed as literals
                return None
            members = [
                self._infer_value_for_default(None, member)
                # Sort by the repr so that the output of stubdefaulter is deterministic,
                # since the ordering of a set at runtime isn't deterministic
                for member in sorted(runtime_default, key=repr)
            ]
            if None in members:
                return None
            return libcst.Set(
                elements=[
                    libcst.Element(cast(libcst.BaseExpression, member))
                    for member in members
                ]
            )
        elif type(runtime_default) is dict:
            infer_default = self._infer_value_for_default
            mapping = {
                infer_default(None, key): infer_default(None, value)
                for key, value in runtime_default.items()
            }
            if None in mapping or None in mapping.values():
                return None
            return libcst.Dict(
                elements=[
                    libcst.DictElement(
                        key=cast(libcst.BaseExpression, key),
                        value=cast(libcst.BaseExpression, value),
                    )
                    for key, value in mapping.items()
                ]
            )
        return None

    def leave_Param(
        self, original_node: libcst.Param, updated_node: libcst.Param
    ) -> libcst.Param:
        if original_node.default is None:
            return updated_node
        inferred_default = self.infer_value_for_default(original_node)
        if inferred_default is None:
            return updated_node
        if isinstance(original_node.default, libcst.Ellipsis):
            self.num_added += 1
            return updated_node.with_changes(default=inferred_default)
        else:
            existing_value = infer_value_of_node(original_node.default)
            if existing_value is NotImplemented:
                return updated_node
            inferred_value = infer_value_of_node(inferred_default)
            if existing_value != inferred_value or type(inferred_value) is not type(
                existing_value
            ):
                self.errors.append(
                    (original_node.name.value, existing_value, inferred_value)
                )
            return updated_node


def get_end_lineno(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if sys.version_info >= (3, 8):
        assert hasattr(node, "end_lineno")
        assert node.end_lineno is not None
        return node.end_lineno
    else:
        return max(
            child.lineno
            for child in ast.iter_child_nodes(node)
            if hasattr(child, "lineno")
        )


def replace_defaults_in_func(
    stub_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    runtime_func: Any,
) -> tuple[int, list[str], dict[int, list[str]]]:
    try:
        sig = inspect.signature(runtime_func)
    except Exception:
        return 0, [], {}
    end_lineno = get_end_lineno(node)
    lines = stub_lines[node.lineno - 1 : end_lineno]
    indentation = len(lines[0]) - len(lines[0].lstrip())
    cst = libcst.parse_statement(
        textwrap.dedent("".join(line + "\n" for line in lines))
    )
    assert isinstance(cst, libcst.FunctionDef)
    visitor = ReplaceEllipsesUsingRuntime(sig, cst.params)
    modified = cst.visit(visitor)
    assert isinstance(modified, libcst.FunctionDef)
    new_code = textwrap.indent(libcst.Module(body=[modified]).code, " " * indentation)
    output_dict = {node.lineno - 1: new_code.splitlines()}
    for i in range(node.lineno, end_lineno):
        output_dict[i] = []
    errors = [
        f"parameter {param_name}: stub default {stub_default!r} != runtime default {runtime_default!r}"
        for param_name, stub_default, runtime_default in visitor.errors
    ]
    return visitor.num_added, errors, output_dict


def gather_funcs(
    node: typeshed_client.NameInfo,
    name: str,
    fullname: str,
    runtime_parent: type | types.ModuleType,
    blacklisted_objects: frozenset[str],
) -> Iterator[Tuple[Union[ast.FunctionDef, ast.AsyncFunctionDef], Any]]:
    if fullname in blacklisted_objects:
        log(f"Skipping {fullname}: blacklisted object")
        return
    interesting_classes = (
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        typeshed_client.OverloadedName,
    )
    if not isinstance(node.ast, interesting_classes):
        return
    # special-case some aliases in the typing module
    if isinstance(runtime_parent, type(typing.Mapping)):
        runtime_parent = runtime_parent.__origin__  # type: ignore[attr-defined]
    try:
        try:
            runtime = getattr(runtime_parent, name)
        except AttributeError:
            runtime = inspect.getattr_static(runtime_parent, name)
    # Some getattr() calls raise TypeError, or something even more exotic
    except Exception:
        log("Could not find", fullname, "at runtime")
        return
    if isinstance(node.ast, ast.ClassDef):
        if not node.child_nodes:
            return
        for child_name, child_node in node.child_nodes.items():
            if child_name.startswith("__") and not child_name.endswith("__"):
                unmangled_parent_name = fullname.split(".")[-1]
                maybe_mangled_child_name = (
                    f"_{unmangled_parent_name.lstrip('_')}{child_name}"
                )
            else:
                maybe_mangled_child_name = child_name
            yield from gather_funcs(
                node=child_node,
                name=maybe_mangled_child_name,
                fullname=f"{fullname}.{child_name}",
                runtime_parent=runtime,
                blacklisted_objects=blacklisted_objects,
            )
    elif isinstance(node.ast, typeshed_client.OverloadedName):
        for definition in node.ast.definitions:
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield definition, runtime
    elif isinstance(node.ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield node.ast, runtime


def add_defaults_to_stub_using_runtime(
    module_name: str,
    context: typeshed_client.finder.SearchContext,
    blacklisted_objects: frozenset[str],
) -> tuple[list[str], int]:
    path = typeshed_client.get_stub_file(module_name, search_context=context)
    if path is None:
        raise ValueError(f"Could not find stub for {module_name}")
    try:
        # Redirect stdout when importing modules to avoid noisy output from modules like `this`
        with contextlib.redirect_stdout(io.StringIO()):
            runtime_module = importlib.import_module(module_name)
    except KeyboardInterrupt:
        raise
    # `importlib.import_module("multiprocessing.popen_fork")` crashes with AttributeError on Windows
    # Trying to import serial.__main__ for typeshed's pyserial package will raise SystemExit
    except BaseException as e:
        log(f'Could not import {module_name}: {type(e).__name__}: "{e}"')
        return [], 0
    stub_names = typeshed_client.get_stub_names(module_name, search_context=context)
    if stub_names is None:
        raise ValueError(f"Could not find stub for {module_name}")
    stub_lines = path.read_text().splitlines()
    # pyanalyze doesn't let you use dict[] here
    replacement_lines: Dict[int, List[str]] = {}
    total_num_added = 0
    errors = []
    for name, info in stub_names.items():
        funcs = gather_funcs(
            node=info,
            name=name,
            fullname=f"{module_name}.{name}",
            runtime_parent=runtime_module,
            blacklisted_objects=blacklisted_objects,
        )

        for func, runtime_func in funcs:
            num_added, new_errors, new_lines = replace_defaults_in_func(
                stub_lines, func, runtime_func
            )
            for error in new_errors:
                message = f"{module_name}.{name}: {error}"
                errors.append(message)
                print(colored(message, "red"))
            replacement_lines.update(new_lines)
            total_num_added += num_added
    with path.open("w") as f:
        for i, line in enumerate(stub_lines):
            if i in replacement_lines:
                for new_line in replacement_lines[i]:
                    f.write(new_line + "\n")
            else:
                f.write(line + "\n")
    return errors, total_num_added


@dataclass
class ReplaceEllipsesUsingAnnotations(libcst.CSTTransformer):
    num_added: int = 0

    @staticmethod
    def node_represents_subscripted_Literal(
        node: libcst.Subscript,
    ) -> bool:
        subscript_value = node.value
        if isinstance(subscript_value, libcst.Name):
            return subscript_value.value == "Literal"
        if isinstance(subscript_value, libcst.Attribute):
            return (
                isinstance(subscript_value.value, libcst.Name)
                and subscript_value.value.value in {"typing", "typing_extensions"}
                and isinstance(subscript_value.attr, libcst.Name)
                and subscript_value.attr.value == "Literal"
            )
        return False

    def leave_Param(
        self, original_node: libcst.Param, updated_node: libcst.Param
    ) -> libcst.Param:
        if not isinstance(original_node.default, libcst.Ellipsis):
            return updated_node
        annotation = original_node.annotation
        if not isinstance(annotation, libcst.Annotation):
            return updated_node
        if (
            isinstance(annotation.annotation, libcst.Name)
            and annotation.annotation.value == "None"
        ):
            self.num_added += 1
            return updated_node.with_changes(default=libcst.Name(value="None"))
        if isinstance(
            annotation.annotation, libcst.Subscript
        ) and self.node_represents_subscripted_Literal(annotation.annotation):
            subscript = annotation.annotation
            if (
                len(subscript.slice) == 1
                and isinstance(subscript.slice[0], libcst.SubscriptElement)
                and isinstance(subscript.slice[0].slice, libcst.Index)
            ):
                literal_slice_contents = subscript.slice[0].slice.value
                if infer_value_of_node(
                    literal_slice_contents
                ) is not NotImplemented and not default_is_too_long(
                    literal_slice_contents
                ):
                    self.num_added += 1
                    return updated_node.with_changes(default=literal_slice_contents)
        return updated_node


def add_defaults_to_stub_using_annotations(
    module_name: str, context: typeshed_client.finder.SearchContext
) -> int:
    path = typeshed_client.get_stub_file(module_name, search_context=context)
    if path is None:
        raise ValueError(f"Could not find stub for {module_name}")
    source = path.read_text()
    cst = libcst.parse_module(source)
    visitor = ReplaceEllipsesUsingAnnotations()
    modified_cst = cst.visit(visitor)
    if visitor.num_added > 0:
        path.write_text(modified_cst.code)
    return visitor.num_added


def add_defaults_to_stub(
    module_name: str,
    context: typeshed_client.finder.SearchContext,
    blacklisted_objects: frozenset[str],
) -> tuple[list[str], int]:
    print(f"Processing {module_name}... ", end="", flush=True)
    num_added_using_annotations = add_defaults_to_stub_using_annotations(
        module_name, context
    )
    errors, num_added_using_runtime = add_defaults_to_stub_using_runtime(
        module_name, context, blacklisted_objects
    )
    total_num_added = num_added_using_annotations + num_added_using_runtime
    print(f"added {total_num_added} defaults")
    return errors, total_num_added


def is_relative_to(left: Path, right: Path) -> bool:
    """Return True if the path is relative to another path or False.

    Redundant with Path.is_relative_to in 3.9+.

    """
    try:
        left.relative_to(right)
        return True
    except ValueError:
        return False


def install_typeshed_packages(typeshed_paths: Sequence[Path]) -> None:
    to_install: List[str] = []
    for path in typeshed_paths:
        metadata_path = path / "METADATA.toml"
        if not metadata_path.exists():
            print(f"{path} does not look like a typeshed package", file=sys.stderr)
            sys.exit(1)
        metadata_bytes = metadata_path.read_text()
        metadata = tomli.loads(metadata_bytes)
        version = metadata["version"]
        to_install.append(f"{path.name}=={version}")
    if to_install:
        command = [sys.executable, "-m", "pip", "install", *to_install]
        print(f"Running install command: {' '.join(command)}")
        subprocess.check_call(command)


# A hardcoded list of stdlib modules to skip
# This is separate to the --blacklists argument on the command line,
# which is for individual functions/methods/variables to skip
#
# `_typeshed` doesn't exist at runtime; no point trying to add defaults
# `antigravity` exists at runtime but it's annoying to have the browser open up every time
STDLIB_MODULE_BLACKLIST = ("_typeshed/*.pyi", "antigravity.pyi")


def load_blacklist(path: Path) -> frozenset[str]:
    with path.open() as f:
        entries = frozenset(line.split("#")[0].strip() for line in f)
    return entries - {""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--stdlib-path",
        help=(
            "Path to typeshed's stdlib directory. If given, we will add defaults to"
            " stubs in this directory."
        ),
    )
    parser.add_argument(
        "-p",
        "--packages",
        nargs="+",
        help=(
            "List of packages to add defaults to. We will add defaults to all stubs in"
            " these directories. The runtime package must be installed."
        ),
        default=(),
    )
    parser.add_argument(
        "-t",
        "--typeshed-packages",
        nargs="+",
        help=(
            "List of typeshed packages to add defaults to. WARNING: We will install the package locally."
        ),
        default=(),
    )
    parser.add_argument(
        "-b",
        "--blacklists",
        nargs="+",
        help=(
            "List of paths pointing to 'blacklist files',"
            " which can be used to specify functions that stubdefaulter should skip"
            " trying to add default values to. Note: if the name of a class is included"
            " in a blacklist, the whole class will be skipped."
        ),
        default=(),
    )
    parser.add_argument(
        "-z",
        "--exit-zero",
        action="store_true",
        help="Exit with code 0 even if there were errors.",
    )
    args = parser.parse_args()

    stdlib_path = Path(args.stdlib_path) if args.stdlib_path else None
    if stdlib_path is not None:
        if not (stdlib_path.is_dir() and (stdlib_path / "VERSIONS").is_file()):
            parser.error(f'"{stdlib_path}" does not point to a valid stdlib directory')

    typeshed_paths = [Path(p) for p in args.typeshed_packages]
    install_typeshed_packages(typeshed_paths)
    package_paths = [Path(p) for p in args.packages] + typeshed_paths
    stdlib_blacklist_path = Path(__file__).parent / "stdlib-blacklist.txt"
    assert stdlib_blacklist_path.exists() and stdlib_blacklist_path.is_file()
    blacklist_paths = [Path(p) for p in args.blacklists] + [stdlib_blacklist_path]

    combined_blacklist = frozenset(
        chain.from_iterable(load_blacklist(path) for path in blacklist_paths)
    )
    context = typeshed_client.finder.get_search_context(
        typeshed=stdlib_path, search_path=package_paths, version=sys.version_info[:2]
    )
    errors = []
    total_num_added = 0
    for module, path in typeshed_client.get_all_stub_files(context):
        if stdlib_path is not None and is_relative_to(path, stdlib_path):
            if any(
                path.relative_to(stdlib_path).match(pattern)
                for pattern in STDLIB_MODULE_BLACKLIST
            ):
                log(f"Skipping {module}: blacklisted module")
                continue
            else:
                these_errors, num_added = add_defaults_to_stub(
                    module, context, combined_blacklist
                )
                errors += these_errors
                total_num_added += num_added
        elif any(is_relative_to(path, p) for p in package_paths):
            these_errors, num_added = add_defaults_to_stub(
                module, context, combined_blacklist
            )
            errors += these_errors
            total_num_added += num_added
    m = f"\n--- Added {total_num_added} defaults; encountered {len(errors)} errors ---"
    print(colored(m, "red" if errors else "green"))
    sys.exit(1 if (errors and not args.exit_zero) else 0)
