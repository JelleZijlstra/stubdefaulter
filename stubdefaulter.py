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
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union

import libcst
import tomli
import typeshed_client
from termcolor import colored


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
    else:
        return NotImplemented


@dataclass
class ReplaceEllipses(libcst.CSTTransformer):
    sig: inspect.Signature
    num_added: int = 0
    errors: List[Tuple[str, object, object]] = field(default_factory=list)

    def infer_value_for_default(
        self, node: libcst.Param
    ) -> libcst.BaseExpression | None:
        param_name = node.name.value
        param: inspect.Parameter | None = None
        try:
            param = self.sig.parameters[param_name]
        except KeyError:
            if param_name.startswith("__") and not param_name.endswith("__"):
                param = self.sig.parameters.get(param_name[2:])
        if not isinstance(param, inspect.Parameter):
            return None
        if param.default is inspect.Parameter.empty:
            return None
        if type(param.default) is bool or param.default is None:
            return libcst.Name(value=str(param.default))
        elif type(param.default) is str:
            return libcst.SimpleString(value=repr(param.default))
        elif type(param.default) is int:
            if (
                node.annotation
                and isinstance(node.annotation.annotation, libcst.Name)
                and node.annotation.annotation.value == "bool"
            ):
                # Skip cases where the type is annotated as bool but the default is an int.
                return None
            if param.default >= 0:
                return libcst.Integer(value=str(param.default))
            else:
                return libcst.UnaryOperation(
                    operator=libcst.Minus(),
                    expression=libcst.Integer(value=str(-param.default)),
                )
        elif type(param.default) is float:
            if not math.isfinite(param.default):
                # Edge cases that it's probably not worth handling
                return None
            # `-0.0 == +0.0`, but we want to keep the sign,
            # so use math.copysign() rather than a comparison with 0
            # to determine whether or not it's a negative float
            if math.copysign(1, param.default) < 0:
                return libcst.UnaryOperation(
                    operator=libcst.Minus(),
                    expression=libcst.Float(value=str(-param.default)),
                )
            else:
                return libcst.Float(value=str(param.default))
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
    visitor = ReplaceEllipses(sig)
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
            yield from gather_funcs(
                node=child_node,
                name=child_name,
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


def add_defaults_to_stub(
    module_name: str,
    context: typeshed_client.finder.SearchContext,
    blacklisted_objects: frozenset[str],
) -> tuple[list[str], int]:
    print(f"Processing {module_name}... ", end="", flush=True)
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
        versions_path = stdlib_path / "VERSIONS"
        if not (stdlib_path.is_dir() and versions_path.is_file()):
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


if __name__ == "__main__":
    main()
