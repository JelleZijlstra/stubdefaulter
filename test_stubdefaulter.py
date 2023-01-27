import sys
import tempfile
from pathlib import Path

import typeshed_client

import stubdefaulter

PY_FILE = """
import enum
import re
import sys

A = 0
B = 'foo'
C = True
D = 5
if sys.version_info >= (3, 5):
    # A test for indented constants:
    E = 'foo'
    F = False

def f(x=0, y="y", z=True, a=None):
    pass
def more_ints(x=-1, y=0):
    pass
def wrong_default(wrong=0):
    pass
def floats(a=1.23456, b=0.0, c=-9.87654, d=-0.0):
    pass
def float_edge_cases(one=float("nan"), two=float("inf"), three=float("-inf")):
    pass

class Capybara:
    def __init__(self, x=0, y="y", z=True, a=None):
        pass
    def overloaded_method(x=False):
        return 1 if x else "1"

class Klass:
    class NestedKlass1:
        class NestedKlass2:
            def method(self, a=False):
                pass
            async def async_method(self, b=3.14):
                pass

def overloaded(x=False):
    return 1 if x else "1"

def intenum_default(x=re.ASCII):
    return int(x)

class FooEnum(str, enum.Enum):
    FOO = "foo"

def strenum_default(x=FooEnum.FOO):
    return str(x)
"""
INPUT_STUB = """
import enum
import re
import sys
from typing import overload, Literal

A: int
B: str
C: bool
D: int = ...
if sys.version_info >= (3, 5):
    # A test for indented constants:
    E: str = ...
    F: bool = ...

def f(x: int = ..., y: str = ..., z: bool = ..., a: Any = ...) -> None: ...
def more_ints(x: int = ..., y: bool = ...) -> None: ...
def wrong_default(wrong: int = 1) -> None: ...
def floats(a: float = ..., b: float = ..., c: float = ..., d: float = ...) -> None: ...
def float_edge_cases(one: float = ..., two: float = ..., three: float = ...) -> None: ...

class Capybara:
    def __init__(self, x: int = ..., y: str = ..., z: bool = ..., a: Any = ...) -> None: ...
    @overload
    def overloaded_method(x: Literal[False] = ...) -> str: ...
    @overload
    def overloaded_method(x: Literal[True]) -> int: ...

class Klass:
    class NestedKlass1:
        class NestedKlass2:
            def method(self, a: bool = ...) -> None: ...
            async def async_method(self, b: float = ...) -> None: ...

@overload
def overloaded(x: Literal[False] = ...) -> str: ...
@overload
def overloaded(x: Literal[True]) -> int: ...

def intenum_default(x: int = ...) -> int: ...

class FooEnum(str, enum.Enum):
    FOO: str

def strenum_default(x: str = ...) -> str: ...
"""
EXPECTED_STUB = """
import enum
import re
import sys
from typing import overload, Literal

A: int = 0
B: str = 'foo'
C: bool = True
D: int = 5
if sys.version_info >= (3, 5):
    # A test for indented constants:
    E: str = 'foo'
    F: bool = False

def f(x: int = 0, y: str = 'y', z: bool = True, a: Any = None) -> None: ...
def more_ints(x: int = -1, y: bool = ...) -> None: ...
def wrong_default(wrong: int = 1) -> None: ...
def floats(a: float = 1.23456, b: float = 0.0, c: float = -9.87654, d: float = -0.0) -> None: ...
def float_edge_cases(one: float = ..., two: float = ..., three: float = ...) -> None: ...

class Capybara:
    def __init__(self, x: int = 0, y: str = 'y', z: bool = True, a: Any = None) -> None: ...
    @overload
    def overloaded_method(x: Literal[False] = False) -> str: ...
    @overload
    def overloaded_method(x: Literal[True]) -> int: ...

class Klass:
    class NestedKlass1:
        class NestedKlass2:
            def method(self, a: bool = False) -> None: ...
            async def async_method(self, b: float = 3.14) -> None: ...

@overload
def overloaded(x: Literal[False] = False) -> str: ...
@overload
def overloaded(x: Literal[True]) -> int: ...

def intenum_default(x: int = ...) -> int: ...

class FooEnum(str, enum.Enum):
    FOO: str

def strenum_default(x: str = ...) -> str: ...
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
