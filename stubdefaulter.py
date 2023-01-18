from __future__ import annotations

"""

Tool to add default values to stubs.

Usage: python stubdefaulter.py path/to/typeshed

TODO:
- Support methods, not just top-level functions
- Maybe enable adding more default values (floats?)

"""

import argparse
import ast
import importlib
import inspect
import itertools
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import libcst
import typeshed_client


def contains_ellipses(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for default in itertools.chain(node.args.defaults, node.args.kw_defaults):
        if isinstance(default, ast.Constant) and default.value is Ellipsis:
            return True
    return False


@dataclass
class ReplaceEllipses(libcst.CSTTransformer):
    sig: inspect.Signature

    def leave_Param(
        self, original_node: libcst.Param, updated_node: libcst.Param
    ) -> libcst.Param:
        if not isinstance(original_node.default, libcst.Ellipsis):
            return updated_node
        try:
            param = self.sig.parameters[original_node.name.value]
        except KeyError:
            return updated_node
        if param.default is inspect.Parameter.empty:
            return updated_node
        if isinstance(param.default, bool) or param.default is None:
            return updated_node.with_changes(
                default=libcst.Name(value=str(param.default))
            )
        elif isinstance(param.default, str):
            return updated_node.with_changes(
                default=libcst.SimpleString(value=repr(param.default))
            )
        elif isinstance(param.default, int) and param.default >= 0:
            if param.default >= 0:
                default = libcst.Integer(value=str(param.default))
            else:
                default = libcst.UnaryOperation(
                    operator=libcst.Minus(),
                    expression=libcst.Integer(value=str(-param.default)),
                )
            return updated_node.with_changes(default=default)
        return updated_node


def replace_defaults_in_func(
    stub_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    runtime_func: Any,
) -> dict[int, list[str]]:
    try:
        sig = inspect.signature(runtime_func)
    except Exception:
        return {}
    assert node.end_lineno is not None
    lines = stub_lines[node.lineno - 1 : node.end_lineno]
    indentation = len(lines[0]) - len(lines[0].lstrip())
    cst = libcst.parse_statement(
        textwrap.dedent("".join(line + "\n" for line in lines))
    )
    modified = cst.visit(ReplaceEllipses(sig))
    assert isinstance(modified, libcst.FunctionDef)
    new_code = textwrap.indent(libcst.Module(body=[modified]).code, " " * indentation)
    output_dict = {node.lineno - 1: new_code.splitlines()}
    for i in range(node.lineno, node.end_lineno):
        output_dict[i] = []
    return output_dict


def add_defaults_to_stub(
    module_name: str, context: typeshed_client.finder.SearchContext
) -> None:
    print(f"Processing {module_name}")
    path = typeshed_client.get_stub_file(module_name, search_context=context)
    if path is None:
        raise ValueError(f"Could not find stub for {module_name}")
    try:
        runtime_module = importlib.import_module(module_name)
    except ImportError:
        print("Could not import", module_name)
        return None
    stub_names = typeshed_client.get_stub_names(module_name, search_context=context)
    if stub_names is None:
        raise ValueError(f"Could not find stub for {module_name}")
    stub_lines = path.read_text().splitlines()
    replacement_lines: dict[int, list[str]] = {}
    for name, info in stub_names.items():
        if isinstance(
            info.ast, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and contains_ellipses(info.ast):
            try:
                runtime_func = getattr(runtime_module, name)
            except AttributeError:
                print("Could not find", name, "in runtime module")
                continue
            replacement_lines.update(
                replace_defaults_in_func(stub_lines, info.ast, runtime_func)
            )
    with path.open("w") as f:
        for i, line in enumerate(stub_lines):
            if i in replacement_lines:
                for new_line in replacement_lines[i]:
                    f.write(new_line + "\n")
            else:
                f.write(line + "\n")


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
    )
    args = parser.parse_args()

    stdlib_path = Path(args.stdlib_path) if args.stdlib_path else None
    package_paths = [Path(p) for p in args.packages]

    context = typeshed_client.finder.get_search_context(
        typeshed=stdlib_path, search_path=package_paths, version=sys.version_info[:2]
    )
    for module, path in typeshed_client.get_all_stub_files(context):
        if stdlib_path is not None and path.is_relative_to(stdlib_path):
            add_defaults_to_stub(module, context)
        elif any(path.is_relative_to(p) for p in package_paths):
            add_defaults_to_stub(module, context)


if __name__ == "__main__":
    main()
