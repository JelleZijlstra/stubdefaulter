"""

Tool to add default values to stubs.

"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import inspect
import io
import math
import shutil
import subprocess
import sys
import textwrap
import types
import typing
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, cast

import libcst.metadata
import tomli
import typeshed_client.finder
import typeshed_client.parser
from libcst.metadata import MetadataWrapper, PositionProvider
from termcolor import colored

# Defaults with a repr longer than this number will not be added.
# This is an arbitrary value, but it's useful to have a cut-off point somewhere.
# There's no real use case for *very* long defaults,
# and they have the potential to cause severe performance issues
# for tools that try to parse or display Python source code
DEFAULT_LENGTH_LIMIT = 500

# Error codes
MISSING_DEFAULT = "missing-default"
WRONG_DEFAULT = "wrong-default"
MISSING_SLOTS = "missing-slots"
DISJOINT_BASE_WITH_SLOTS = "disjoint-base-with-slots"
ALL_ERROR_CODES = frozenset(
    {MISSING_DEFAULT, WRONG_DEFAULT, MISSING_SLOTS, DISJOINT_BASE_WITH_SLOTS}
)


@dataclass(frozen=True)
class Config:
    """Configuration for the stubdefaulter tool."""

    add_complex_defaults: bool
    enabled_errors: frozenset[str]
    apply_fixes: bool
    blacklisted_objects: frozenset[str]
    verbose: bool = False


@dataclass
class LintError:
    code: str
    message: str
    filename: str
    line: int
    fixed: bool
    source_lines: Sequence[str]
    is_emitted: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.is_emitted = not is_suppressed(
            self.code, line=self.line, source_lines=self.source_lines
        )
        if not self.is_emitted:
            self.fixed = False


def default_is_too_long(default: libcst.BaseExpression) -> bool:
    default_as_module = libcst.Module(
        body=[libcst.SimpleStatementLine(body=[libcst.Expr(value=default)])]
    )
    repr_of_default = default_as_module.code.strip()
    return len(repr_of_default) > DEFAULT_LENGTH_LIMIT


def log(config: Config, *objects: object) -> None:
    if config.verbose:
        print(colored(" ".join(map(str, objects)), "yellow"))


def is_suppressed(code: str, *, line: int, source_lines: Sequence[str]) -> bool:
    """Return True if the given error code is suppressed on the line.

    Suppression markers (case-insensitive):
    - "# stubdefaulter: ignore[<code>]"
    - "# noqa: <code>"
    """
    if line <= 0 or line > len(source_lines):
        return False
    lt = source_lines[line - 1].lower()
    if f"stubdefaulter: ignore[{code}]" in lt:
        return True
    if f"noqa: {code}" in lt:
        return True
    return False


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


def is_complex_default(value: object, *, allow_containers: bool = True) -> bool:
    # Mostly flake8-pyi Y015
    if isinstance(value, (str, bytes)):
        return len(value) > 50  # flake8-pyi Y053
    elif isinstance(value, (int, float, complex)) or value is None or value is ...:
        return len(str(value)) > 10  # flake8-pyi Y054
    elif isinstance(value, (list, tuple, set)):
        return (
            (not allow_containers)
            or len(value) > 10
            or any(is_complex_default(item, allow_containers=False) for item in value)
        )
    elif isinstance(value, dict):
        return (
            not allow_containers
            or len(value) > 10
            or any(
                is_complex_default(key, allow_containers=False)
                or is_complex_default(value, allow_containers=False)
                for key, value in value.items()
            )
        )
    else:
        return True


def get_position(
    visitor: libcst.CSTTransformer, node: libcst.CSTNode
) -> libcst.metadata.CodeRange:
    """Get the position of a node using metadata."""
    pos = visitor.get_metadata(PositionProvider, node)
    if not isinstance(pos, libcst.metadata.CodeRange):
        raise ValueError("Node has no position metadata")
    return pos


@dataclass
class ReplaceEllipsesUsingRuntime(libcst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)
    sig: inspect.Signature
    stub_params: libcst.Parameters
    config: Config
    file_path: str
    base_line_offset: int
    source_lines: Sequence[str]
    errors: list[LintError] = field(default_factory=list)

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
        if (not self.config.add_complex_defaults) and is_complex_default(param.default):
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
            libcst_cls: type[libcst.Tuple | libcst.List]
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
        param_name = original_node.name.value
        if isinstance(original_node.default, libcst.Ellipsis):
            if MISSING_DEFAULT in self.config.enabled_errors:
                runtime_value = infer_value_of_node(inferred_default)
                pos = get_position(self, original_node)
                abs_line = self.base_line_offset + pos.start.line
                error = LintError(
                    MISSING_DEFAULT,
                    f"parameter {param_name} missing default {runtime_value!r}",
                    filename=self.file_path,
                    line=abs_line,
                    fixed=self.config.apply_fixes,
                    source_lines=self.source_lines,
                )
                self.errors.append(error)
                if error.is_emitted:
                    return updated_node.with_changes(default=inferred_default)
            return updated_node
        else:
            existing_value = infer_value_of_node(original_node.default)
            if existing_value is NotImplemented:
                return updated_node
            inferred_value = infer_value_of_node(inferred_default)
            if existing_value != inferred_value or type(inferred_value) is not type(
                existing_value
            ):
                if WRONG_DEFAULT in self.config.enabled_errors:
                    pos = get_position(self, original_node)
                    abs_line = self.base_line_offset + pos.start.line
                    error = LintError(
                        WRONG_DEFAULT,
                        f"parameter {param_name}: stub default {existing_value!r} != runtime default {inferred_value!r}",
                        filename=self.file_path,
                        line=abs_line,
                        fixed=self.config.apply_fixes,
                        source_lines=self.source_lines,
                    )
                    self.errors.append(error)
                    if error.is_emitted:
                        return updated_node.with_changes(default=inferred_default)
            return updated_node


def get_end_lineno(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.expr) -> int:
    assert node.end_lineno is not None
    return node.end_lineno


def get_start_lineno(node: ast.AST) -> int:
    linenos: list[int] = []
    for subnode in ast.walk(node):
        lineno = getattr(subnode, "lineno", None)
        if lineno is not None:
            linenos.append(lineno)
    if not linenos:
        raise ValueError("Node has no lineno attribute")
    return min(linenos)


def replace_defaults_in_func(
    stub_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    runtime_func: Any,
    *,
    config: Config,
    path: Path,
) -> tuple[list[LintError], dict[int, list[str]]]:
    try:
        sig = inspect.signature(runtime_func)
    except Exception:
        return [], {}
    end_lineno = get_end_lineno(node)
    lines = stub_lines[node.lineno - 1 : end_lineno]
    indentation = len(lines[0]) - len(lines[0].lstrip())
    cst = libcst.parse_statement(
        textwrap.dedent("".join(line + "\n" for line in lines))
    )
    assert isinstance(cst, libcst.FunctionDef)
    # Wrap a synthetic module to enable metadata and transformations
    module = libcst.Module(body=[cst])
    wrapper = MetadataWrapper(module, unsafe_skip_copy=True)
    visitor = ReplaceEllipsesUsingRuntime(
        sig,
        cst.params,
        config=config,
        file_path=str(path),
        base_line_offset=node.lineno - 1,
        source_lines=stub_lines,
    )
    modified_module = wrapper.visit(visitor)
    output_dict: dict[int, list[str]] = {}
    if config.apply_fixes and any(error.fixed for error in visitor.errors):
        new_code = textwrap.indent(modified_module.code, " " * indentation)
        output_dict = {node.lineno - 1: new_code.splitlines()}
        for i in range(node.lineno, end_lineno):
            output_dict[i] = []
    return visitor.errors, output_dict


def is_ellipsis_stmt(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and node.value.value is Ellipsis
    )


def is_docstring_stmt(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def is_disjoint_base_decorator(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == "disjoint_base") or (
        isinstance(node, ast.Attribute)
        and node.attr == "disjoint_base"
        and (
            isinstance(node.value, ast.Name)
            and node.value.id in ("typing_extensions", "typing")
        )
    )


def remove_redundant_disjoint_base(
    info: typeshed_client.NameInfo,
    replacement_lines: dict[int, list[str]],
    *,
    config: Config,
    qualname: str,
    path: Path,
    stub_lines: Sequence[str],
) -> Iterable[LintError]:
    node = info.ast
    assert isinstance(node, ast.ClassDef), "Expected node to be a ClassDef"
    if not node.decorator_list or info.child_nodes is None:
        return
    if "__slots__" not in info.child_nodes:
        return
    for deco in node.decorator_list:
        if is_disjoint_base_decorator(deco):
            error = LintError(
                DISJOINT_BASE_WITH_SLOTS,
                f"{qualname} has disjoint_base decorator, but also has __slots__",
                filename=str(path),
                line=deco.lineno,
                fixed=bool(config.apply_fixes),
                source_lines=stub_lines,
            )
            yield error
            if not error.is_emitted:
                return

            lines_to_remove = range(get_start_lineno(deco) - 1, get_end_lineno(deco))
            for lineno in lines_to_remove:
                replacement_lines[lineno] = []


def add_slots_to_class(
    stub_lines: list[str],
    info: typeshed_client.NameInfo,
    runtime_cls: type,
    replacement_lines: dict[int, list[str]],
    *,
    config: Config,
    qualname: str,
    path: Path,
) -> Iterable[LintError]:
    runtime_slots = runtime_cls.__dict__.get("__slots__")
    if runtime_slots is None:
        return
    if isinstance(getattr(runtime_cls, "_fields", None), tuple):
        # Probably a namedtuple, which always have empty __slots__. Not interesting.
        return
    if info.child_nodes is not None and "__slots__" in info.child_nodes:
        return

    node = info.ast
    assert isinstance(node, ast.ClassDef), "Expected node to be a ClassDef"

    error = LintError(
        MISSING_SLOTS,
        f"{qualname} missing __slots__",
        filename=str(path),
        line=node.lineno,
        fixed=bool(config.apply_fixes),
        source_lines=stub_lines,
    )
    yield error
    if not error.is_emitted:
        return

    indentation = (
        len(stub_lines[node.lineno - 1]) - len(stub_lines[node.lineno - 1].lstrip()) + 4
    )
    new_line = " " * indentation + f"__slots__ = {repr(runtime_slots)}"

    if len(node.body) == 1 and is_ellipsis_stmt(node.body[0]):
        line_index = get_start_lineno(node.body[0]) - 1
        class_line = stub_lines[line_index]
        header = class_line.split(":")[0] + ":"
        replacement_lines[line_index] = [header, new_line]
        return

    if node.body and is_docstring_stmt(node.body[0]):
        doc = node.body[0]
        assert doc.end_lineno is not None
        line_index = doc.end_lineno
    elif node.body:
        line_index = get_start_lineno(node.body[0]) - 1
    else:
        line_index = node.lineno

    if line_index in replacement_lines:
        replacement_lines[line_index] = [new_line, *replacement_lines[line_index]]
    else:
        if line_index < len(stub_lines):
            replacement_lines[line_index] = [new_line, stub_lines[line_index]]
        else:
            replacement_lines[line_index] = [new_line]


def gather_funcs(
    node: typeshed_client.NameInfo,
    name: str,
    fullname: str,
    runtime_parent: type | types.ModuleType,
    blacklisted_objects: frozenset[str],
    *,
    config: Config,
) -> Iterator[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Any]]:
    if fullname in blacklisted_objects:
        log(config, f"Skipping {fullname}: blacklisted object")
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
        log(config, "Could not find", fullname, "at runtime")
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
                config=config,
            )
    elif isinstance(node.ast, typeshed_client.OverloadedName):
        for definition in node.ast.definitions:
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield definition, runtime
    elif isinstance(node.ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield node.ast, runtime


def locate_class(
    node: typeshed_client.NameInfo,
    name: str,
    fullname: str,
    runtime_parent: object,
    blacklisted_objects: frozenset[str],
    *,
    config: Config,
) -> type[object] | None:
    if fullname in blacklisted_objects:
        log(config, f"Skipping {fullname}: blacklisted object")
        return None
    if not isinstance(node.ast, ast.ClassDef):
        return None
    runtime_parent = getattr(runtime_parent, "__origin__", runtime_parent)
    try:
        try:
            runtime = getattr(runtime_parent, name)
        except AttributeError:
            runtime = inspect.getattr_static(runtime_parent, name)
    except Exception:
        log(config, "Could not find", fullname, "at runtime")
        return None
    if isinstance(runtime, type):
        return runtime
    return None


def visit_classes_without_runtime(
    name_dict: typeshed_client.parser.NameDict,
    stub_lines: list[str],
    replacement_lines: dict[int, list[str]],
    config: Config,
    path: Path,
    enclosing_name: str,
) -> Iterable[LintError]:
    for name, info in name_dict.items():
        if not isinstance(info.ast, ast.ClassDef):
            continue
        qualname = f"{enclosing_name}.{name}"
        yield from remove_redundant_disjoint_base(
            info,
            replacement_lines,
            config=config,
            qualname=qualname,
            path=path,
            stub_lines=stub_lines,
        )
        if info.child_nodes is not None:
            yield from visit_classes_without_runtime(
                info.child_nodes,
                stub_lines,
                replacement_lines,
                config=config,
                enclosing_name=qualname,
                path=path,
            )


def visit_classes_with_runtime(
    name_dict: typeshed_client.parser.NameDict,
    stub_lines: list[str],
    replacement_lines: dict[int, list[str]],
    config: Config,
    path: Path,
    enclosing_name: str,
    runtime_parent: object,
) -> Iterable[LintError]:
    for name, info in name_dict.items():
        if not isinstance(info.ast, ast.ClassDef):
            continue
        qualname = f"{enclosing_name}.{name}"
        runtime_cls = locate_class(
            node=info,
            name=name,
            fullname=qualname,
            runtime_parent=runtime_parent,
            blacklisted_objects=config.blacklisted_objects,
            config=config,
        )
        if runtime_cls is None:
            continue
        if MISSING_SLOTS in config.enabled_errors:
            yield from add_slots_to_class(
                stub_lines,
                info,
                runtime_cls,
                replacement_lines,
                config=config,
                qualname=qualname,
                path=path,
            )
        if DISJOINT_BASE_WITH_SLOTS in config.enabled_errors:
            yield from remove_redundant_disjoint_base(
                info,
                replacement_lines,
                config=config,
                qualname=qualname,
                path=path,
                stub_lines=stub_lines,
            )
        if info.child_nodes is not None:
            yield from visit_classes_with_runtime(
                info.child_nodes,
                stub_lines,
                replacement_lines,
                config=config,
                runtime_parent=runtime_parent,
                enclosing_name=qualname,
                path=path,
            )


def run_checks_with_runtime(
    module_name: str, context: typeshed_client.finder.SearchContext, *, config: Config
) -> Iterable[LintError]:
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
        log(config, f'Could not import {module_name}: {type(e).__name__}: "{e}"')
        return
    stub_names = typeshed_client.get_stub_names(module_name, search_context=context)
    if stub_names is None:
        raise ValueError(f"Could not find stub for {module_name}")
    stub_lines = path.read_text(encoding="utf-8").splitlines()
    replacement_lines: dict[int, list[str]] = {}
    for name, info in stub_names.items():
        funcs = gather_funcs(
            node=info,
            name=name,
            fullname=f"{module_name}.{name}",
            runtime_parent=runtime_module,
            blacklisted_objects=config.blacklisted_objects,
            config=config,
        )

        for func, runtime_func in funcs:
            new_errors, new_lines = replace_defaults_in_func(
                stub_lines, func, runtime_func, config=config, path=path
            )
            yield from new_errors
            if new_lines:
                replacement_lines.update(new_lines)
    if MISSING_SLOTS in config.enabled_errors:
        yield from visit_classes_with_runtime(
            name_dict=stub_names,
            stub_lines=stub_lines,
            replacement_lines=replacement_lines,
            config=config,
            path=path,
            enclosing_name=module_name,
            runtime_parent=runtime_module,
        )
    if DISJOINT_BASE_WITH_SLOTS in config.enabled_errors:
        yield from visit_classes_without_runtime(
            name_dict=stub_names,
            stub_lines=stub_lines,
            replacement_lines=replacement_lines,
            config=config,
            path=path,
            enclosing_name=module_name,
        )
    if config.apply_fixes and replacement_lines:
        with path.open("w", encoding="utf-8") as f:
            for i, line in enumerate(stub_lines):
                if i in replacement_lines:
                    for new_line in replacement_lines[i]:
                        f.write(new_line + "\n")
                else:
                    f.write(line + "\n")


@dataclass
class StubOnlyVisitor(libcst.CSTTransformer):
    METADATA_DEPENDENCIES = (PositionProvider,)
    config: Config
    file_path: str
    source_lines: Sequence[str]
    errors: list[LintError] = field(default_factory=list)

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
        new_default: libcst.BaseExpression | None = None
        if (
            isinstance(annotation.annotation, libcst.Name)
            and annotation.annotation.value == "None"
        ):
            new_default = libcst.Name(value="None")
        elif isinstance(
            annotation.annotation, libcst.Subscript
        ) and self.node_represents_subscripted_Literal(annotation.annotation):
            subscript = annotation.annotation
            if len(subscript.slice) == 1 and isinstance(
                subscript.slice[0].slice, libcst.Index
            ):
                literal_slice_contents = subscript.slice[0].slice.value
                if infer_value_of_node(
                    literal_slice_contents
                ) is not NotImplemented and not default_is_too_long(
                    literal_slice_contents
                ):
                    new_default = literal_slice_contents
        if new_default is None:
            return updated_node
        if MISSING_DEFAULT in self.config.enabled_errors:
            runtime_value = infer_value_of_node(new_default)
            pos = get_position(self, original_node)
            error = LintError(
                MISSING_DEFAULT,
                f"parameter {original_node.name.value} missing default {runtime_value!r}",
                filename=self.file_path,
                line=pos.start.line,
                fixed=self.config.apply_fixes,
                source_lines=self.source_lines,
            )
            self.errors.append(error)
            if error.is_emitted:
                return updated_node.with_changes(default=new_default)
        return updated_node


def run_checks_without_runtime(
    module_name: str,
    context: typeshed_client.finder.SearchContext,
    *,
    config: Config,
) -> list[LintError]:
    path = typeshed_client.get_stub_file(module_name, search_context=context)
    if path is None:
        raise ValueError(f"Could not find stub for {module_name}")
    source = path.read_text(encoding="utf-8")
    cst = libcst.parse_module(source)
    wrapper = MetadataWrapper(cst)
    visitor = StubOnlyVisitor(
        config=config, file_path=str(path), source_lines=source.splitlines()
    )
    modified_cst = wrapper.visit(visitor)
    if config.apply_fixes and any(error.fixed for error in visitor.errors):
        path.write_text(modified_cst.code, encoding="utf-8")
    return visitor.errors


def run_on_stub(
    module_name: str,
    context: typeshed_client.finder.SearchContext,
    *,
    config: Config,
) -> Iterable[LintError]:
    yield from run_checks_without_runtime(module_name, context, config=config)
    yield from run_checks_with_runtime(module_name, context, config=config)


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
    to_install: list[str] = []
    for path in typeshed_paths:
        metadata_path = path / "METADATA.toml"
        if not metadata_path.exists():
            print(f"{path} does not look like a typeshed package", file=sys.stderr)
            sys.exit(1)
        metadata_bytes = metadata_path.read_text(encoding="utf-8")
        metadata = tomli.loads(metadata_bytes)
        version = metadata["version"]
        if version[0] == "~":
            to_install.append(f"{path.name}{version}")
        else:
            to_install.append(f"{path.name}=={version}")
    if to_install:
        if shutil.which("uv") is not None:
            # Use uv to install packages if available
            command = ["uv", "pip", "install", *to_install, "--python", sys.executable]
        else:
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
    with path.open(encoding="utf-8") as f:
        entries = frozenset(line.split("#")[0].strip() for line in f)
    return entries - {""}


def gather_blacklists(paths: Sequence[Path]) -> frozenset[str]:
    combined_blacklist: set[str] = set()
    for path in paths:
        if not path.exists() or not path.is_file():
            raise ValueError(f"Blacklist path {path} does not exist or is not a file")
        combined_blacklist.update(load_blacklist(path))
    return frozenset(combined_blacklist)


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
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit with code 2 when changes where made.",
    )
    parser.add_argument(
        "--add-complex-defaults",
        action="store_true",
        help=(
            "Add complex defaults that are not allowed by typeshed's default linting settings."
        ),
    )
    parser.add_argument(
        "--disable",
        nargs="*",
        choices=sorted(ALL_ERROR_CODES),
        default=[],
        help="Disable specific error codes",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=sorted(ALL_ERROR_CODES),
        default=None,
        help="Enable only the specified error codes (disables all others)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply autofixes to stubs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    stdlib_path = Path(args.stdlib_path) if args.stdlib_path else None
    if stdlib_path is not None:
        if not (stdlib_path.is_dir() and (stdlib_path / "VERSIONS").is_file()):
            parser.error(f'"{stdlib_path}" does not point to a valid stdlib directory')

    typeshed_paths = [Path(p) for p in args.typeshed_packages]
    install_typeshed_packages(typeshed_paths)
    package_paths = [Path(p) for p in args.packages] + typeshed_paths

    blacklist_paths = [Path(p) for p in args.blacklists] + [
        Path(__file__).parent / "stdlib-blacklist.txt",
        Path(__file__).parent / "typeshed-blacklist.txt",
    ]
    combined_blacklist = gather_blacklists(blacklist_paths)

    context = typeshed_client.finder.get_search_context(
        typeshed=stdlib_path, search_path=package_paths, version=sys.version_info[:2]
    )
    if args.only is not None and args.disable:
        parser.error("Cannot use --only with --disable; choose one")
    if args.only is not None:
        enabled_errors = frozenset(args.only)
    else:
        enabled_errors = frozenset(ALL_ERROR_CODES - set(args.disable))
    errors: list[LintError] = []
    config = Config(
        enabled_errors=enabled_errors,
        apply_fixes=args.fix,
        add_complex_defaults=args.add_complex_defaults,
        blacklisted_objects=combined_blacklist,
        verbose=bool(args.verbose),
    )
    # Counts removed; rely on collected errors and their `fixed` status
    for module, path in typeshed_client.get_all_stub_files(context):
        if stdlib_path is not None and is_relative_to(path, stdlib_path):
            if any(
                path.relative_to(stdlib_path).match(pattern)
                for pattern in STDLIB_MODULE_BLACKLIST
            ):
                log(config, f"Skipping {module}: blacklisted module")
                continue
            else:
                errors += run_on_stub(module, context, config=config)

        elif any(is_relative_to(path, p) for p in package_paths):
            errors += run_on_stub(module, context, config=config)

    # Print collected lint results: green if fixed, red if not
    for e in errors:
        if e.is_emitted:
            text = f"{e.filename}:{e.line}: [{e.code}] {e.message}"
            print(colored(text, "green" if e.fixed else "red"))
    # Print summary per error code
    print("\nSummary:")
    for code in sorted(enabled_errors):
        total = sum(1 for err in errors if err.code == code)
        fixed = sum(1 for err in errors if err.code == code and err.fixed)
        suppressed = sum(1 for err in errors if err.code == code and not err.is_emitted)
        print(f"- {code}: {total} errors ({fixed} fixed, {suppressed} suppressed)")

    # Determine exit code
    changed = any(e.fixed for e in errors)
    if args.check and changed:
        exit_code = 2
    elif errors and not args.exit_zero:
        exit_code = 1
    else:
        exit_code = 0
    sys.exit(exit_code)
