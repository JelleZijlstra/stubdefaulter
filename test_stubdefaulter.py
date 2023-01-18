import sys
import tempfile
from pathlib import Path

import typeshed_client

import stubdefaulter

PY_FILE = """
def f(x=0, y="y", z=True, a=None):
    pass
def more_ints(x=-1, y=0):
    pass
def wrong_default(wrong=0):
    pass
class Capybara:
    def __init__(self, x=0, y="y", z=True, a=None):
        pass
"""
INPUT_STUB = """
def f(x: int = ..., y: str = ..., z: bool = ..., a: Any = ...) -> None: ...
def more_ints(x: int = ..., y: bool = ...) -> None: ...
def wrong_default(wrong: int = 1) -> None: ...
class Capybara:
    def __init__(self, x: int = ..., y: str = ..., z: bool = ..., a: Any = ...) -> None: ...
"""
EXPECTED_STUB = """
def f(x: int = 0, y: str = 'y', z: bool = True, a: Any = None) -> None: ...
def more_ints(x: int = -1, y: bool = ...) -> None: ...
def wrong_default(wrong: int = 1) -> None: ...
class Capybara:
    def __init__(self, x: int = 0, y: str = 'y', z: bool = True, a: Any = None) -> None: ...
"""
PKG_NAME = "pkg"


def test_stubdefaulter() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        sys.path.append(tmpdir)
        td = Path(tmpdir)
        pkg_path = td / PKG_NAME
        pkg_path.mkdir()
        stub_path = pkg_path / "__init__.pyi"
        stub_path.write_text(INPUT_STUB)
        (pkg_path / "__init__.py").write_text(PY_FILE)
        (pkg_path / "py.typed").write_text("typed\n")

        errors = stubdefaulter.add_defaults_to_stub(
            PKG_NAME, typeshed_client.finder.get_search_context(search_path=[td])
        )
        assert stub_path.read_text() == EXPECTED_STUB
        assert len(errors) == 1

        stub_path.write_text(INPUT_STUB.replace(" = 1", " = ..."))
        errors = stubdefaulter.add_defaults_to_stub(
            PKG_NAME, typeshed_client.finder.get_search_context(search_path=[td])
        )
        assert stub_path.read_text() == EXPECTED_STUB.replace(
            "wrong: int = 1", "wrong: int = 0"
        )
        assert len(errors) == 0
