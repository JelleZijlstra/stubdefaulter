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
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import libcst
import tomli
import typeshed_client


def infer_value_of_node(node: libcst.BaseExpression) -> object:
    """Return NotImplemented if we can't infer the value."""
    if isinstance(node, libcst.Integer):
        return int(node.value)
    elif isinstance(node, libcst.SimpleString):
        return ast.literal_eval(node.value)
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
            if not isinstance(operand, int):
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

    @staticmethod
    def annotation_is_bool(annotation: libcst.Annotation | None) -> bool:
        return bool(
            annotation
            and isinstance(annotation.annotation, libcst.Name)
            and annotation.annotation.value == "bool"
        )

    def infer_value_for_default(
        self, node: libcst.Param
    ) -> libcst.BaseExpression | None:
        try:
            param = self.sig.parameters[node.name.value]
        except KeyError:
            return None
        if param.default is inspect.Parameter.empty:
            return None
        if type(param.default) is bool or param.default is None:
            return libcst.Name(value=str(param.default))
        elif type(param.default) is str:
            return libcst.SimpleString(value=repr(param.default))
        elif type(param.default) is int:
            if self.annotation_is_bool(node.annotation):
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
            if self.annotation_is_bool(node.annotation):
                # Skip cases where the type is annotated as bool but the default is a float.
                return None
            if str(param.default) in {"nan", "inf", "-inf"}:
                # Edge cases that it's probably not worth handling
                return None
            # `-0.0 == +0.0`, but we want to keep the sign,
            # so use the string representation rather than the value itself
            # to determine whether or not it's a negative float
            if str(param.default).startswith("-"):
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


def add_defaults_to_stub(
    module_name: str, context: typeshed_client.finder.SearchContext
) -> list[str]:
    print(f"Processing {module_name}... ", end="", flush=True)
    path = typeshed_client.get_stub_file(module_name, search_context=context)
    if path is None:
        raise ValueError(f"Could not find stub for {module_name}")
    try:
        # Redirect stdout when importing modules to avoid noisy output from modules like `this`
        with contextlib.redirect_stdout(io.StringIO()):
            runtime_module = importlib.import_module(module_name)
    # `importlib.import_module("multiprocessing.popen_fork")` crashes with AttributeError on Windows
    except Exception as e:
        print(f'Could not import {module_name}: {type(e).__name__}: "{e}"')
        return []
    stub_names = typeshed_client.get_stub_names(module_name, search_context=context)
    if stub_names is None:
        raise ValueError(f"Could not find stub for {module_name}")
    stub_lines = path.read_text().splitlines()
    # pyanalyze doesn't let you use dict[] here
    replacement_lines: Dict[int, List[str]] = {}
    total_num_added = 0
    errors = []
    for name, info in stub_names.items():
        if isinstance(info.ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                runtime_func = getattr(runtime_module, name)
            except AttributeError:
                print("Could not find", name, "in runtime module")
                continue
            funcs = [(info.ast, runtime_func)]
        elif isinstance(info.ast, ast.ClassDef) and info.child_nodes:
            funcs = []
            try:
                runtime_class = getattr(runtime_module, name)
            except AttributeError:
                print("Could not find", name, "in runtime module")
                continue
            for child_name, child_info in info.child_nodes.items():
                if isinstance(child_info.ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    try:
                        runtime_func = getattr(runtime_class, child_name)
                    except AttributeError:
                        print(f"Could not find {name}.{child_name} in runtime module")
                        continue
                    funcs.append((child_info.ast, runtime_func))
        else:
            funcs = []

        for func, runtime_func in funcs:
            num_added, new_errors, new_lines = replace_defaults_in_func(
                stub_lines, func, runtime_func
            )
            for error in new_errors:
                message = f"{module_name}.{name}: {error}"
                errors.append(message)
                print(message)
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
    return errors


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


# `_typeshed` doesn't exist at runtime; no point trying to add defaults
# `antigravity` exists at runtime but it's annoying to have the browser open up every time
STDLIB_MODULE_BLACKLIST = ("_typeshed/*.pyi", "antigravity.pyi")


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
    args = parser.parse_args()

    stdlib_path = Path(args.stdlib_path) if args.stdlib_path else None
    typeshed_paths = [Path(p) for p in args.typeshed_packages]
    install_typeshed_packages(typeshed_paths)
    package_paths = [Path(p) for p in args.packages] + typeshed_paths

    context = typeshed_client.finder.get_search_context(
        typeshed=stdlib_path, search_path=package_paths, version=sys.version_info[:2]
    )
    errors = []
    for module, path in typeshed_client.get_all_stub_files(context):
        if stdlib_path is not None and is_relative_to(path, stdlib_path):
            if any(
                path.relative_to(stdlib_path).match(pattern)
                for pattern in STDLIB_MODULE_BLACKLIST
            ):
                print(f"Skipping {module}: blacklisted module")
                continue
            else:
                errors += add_defaults_to_stub(module, context)
        elif any(is_relative_to(path, p) for p in package_paths):
            errors += add_defaults_to_stub(module, context)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
