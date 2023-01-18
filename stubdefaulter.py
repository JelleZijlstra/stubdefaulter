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
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import libcst
import tomli
import typeshed_client


def contains_ellipses(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for default in itertools.chain(node.args.defaults, node.args.kw_defaults):
        if isinstance(default, ast.Constant) and default.value is Ellipsis:
            return True
    return False


@dataclass
class ReplaceEllipses(libcst.CSTTransformer):
    sig: inspect.Signature
    num_added: int = 0

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
            self.num_added += 1
            return updated_node.with_changes(
                default=libcst.Name(value=str(param.default))
            )
        elif isinstance(param.default, str):
            self.num_added += 1
            return updated_node.with_changes(
                default=libcst.SimpleString(value=repr(param.default))
            )
        elif isinstance(param.default, int) and param.default >= 0:
            self.num_added += 1
            if param.default >= 0:
                default = libcst.Integer(value=str(param.default))
            else:
                default = libcst.UnaryOperation(
                    operator=libcst.Minus(),
                    expression=libcst.Integer(value=str(-param.default)),
                )
            return updated_node.with_changes(default=default)
        return updated_node


def get_end_lineno(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if sys.version_info >= (3, 8):
        assert hasattr(node, "end_lineno")
        assert node.end_lineno is not None
        return node.end_lineno
    else:
        return max(child.lineno for child in ast.iter_child_nodes(node))


def replace_defaults_in_func(
    stub_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    runtime_func: Any,
) -> tuple[int, dict[int, list[str]]]:
    try:
        sig = inspect.signature(runtime_func)
    except Exception:
        return 0, {}
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
    return visitor.num_added, output_dict


def add_defaults_to_stub(
    module_name: str, context: typeshed_client.finder.SearchContext
) -> None:
    print(f"Processing {module_name}... ", end="", flush=True)
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
    # pyanalyze doesn't let you use dict[] here
    replacement_lines: Dict[int, List[str]] = {}
    total_num_added = 0
    for name, info in stub_names.items():
        if isinstance(
            info.ast, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and contains_ellipses(info.ast):
            try:
                runtime_func = getattr(runtime_module, name)
            except AttributeError:
                print("Could not find", name, "in runtime module")
                continue
            num_added, new_lines = replace_defaults_in_func(
                stub_lines, info.ast, runtime_func
            )
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
    for module, path in typeshed_client.get_all_stub_files(context):
        if stdlib_path is not None and is_relative_to(path, stdlib_path):
            add_defaults_to_stub(module, context)
        elif any(is_relative_to(path, p) for p in package_paths):
            add_defaults_to_stub(module, context)


if __name__ == "__main__":
    main()
