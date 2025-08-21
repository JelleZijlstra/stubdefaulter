import sys
import tempfile
from pathlib import Path

import libcst
import pytest
import typeshed_client

import stubdefaulter

PY_FILE = """
import enum
import inspect
import re
import stubdefaulter

def f(x=0, y="y", z=True, a=None):
    pass
def more_ints(x=-1, y=0):
    pass
def ints_as_hexadecimals(x=0x7FFFFFFF, y=0b1101, z=0o744):
    pass
def wrong_default(wrong=0):
    pass
def floats(a=1.23456, b=0.0, c=-9.87654, d=-0.0):
    pass
def float_edge_cases(one=float("nan"), two=float("inf"), three=float("-inf")):
    pass
def bytes_func(one=b"foo"):
    pass
def containers(
    a=(),
    b=[],
    c={},
    d=(1, "foo", b"bar", True, None, 1.23),
    e=[1, "foo", b"bar", True, None, 1.23],
    f={1, "foo", b"bar", False, None, 1.23},
    g={-1: 1, "foo": "foo", b"bar": "bar", True: False, None: None, 1.23: 1.234},
):
    pass
def bad_container(x=set()):
    pass

class Capybara:
    def __init__(self, x=0, y="y", z=True, a=None):
        pass
    def overloaded_method(x=False):
        return 1 if x else "1"
    def __mangled(x="foo"):
        pass

class Slotty:
    __slots__ = ("x", "y")

class Klass:
    class NestedKlass1:
        class NestedKlass2:
            def method(self, a=False):
                pass
            async def async_method(self, b=3.14):
                pass
    class __Mangled1:
        class __Mangled2:
            def __mangled(self, x=True):
                pass

def overloaded(x=False):
    return 1 if x else "1"

def intenum_default(x=re.ASCII):
    return int(x)

class FooEnum(str, enum.Enum):
    FOO = "foo"

def strenum_default(x=FooEnum.FOO):
    return str(x)

def pos_only(x=5):
    pass

def useless_runtime(*args, **kwargs):
    if 'foo' in kwargs and kwargs['foo'] is not None:
        raise TypeError("Passing a non-None value for 'foo' is not allowed")
    if 'bar' in kwargs and kwargs['bar'] != 0:
        raise ValueError("Passing a non-0 value for 'bar' is not allowed")
    for kwarg in ('spam', 'eggs', 'ham'):
        if kwarg in kwargs and kwargs[kwarg] != "foo":
            raise ValueError(
                f"Passing a non-foo value for {kwarg!r}, what are you thinking??"
            )
    if 'enum_default' in kwargs and kwargs['enum_default'] is not re.ASCII:
        raise ValueError("FOOL")

def incorrect_non_posonly_parameter_names_in_stub(x="foo", y="bar", *, z="baz"):
    pass

# Some slightly hacky signature modification is required here,
# so that we can get a pos-only parameter at runtime,
# while having a test that passes on Python 3.7
def incorrect_posonly_parameter_names_in_sub(x="foo"):
    pass
incorrect_posonly_parameter_names_in_sub.__signature__ = inspect.Signature(
    [inspect.Parameter("x", kind=inspect.Parameter.POSITIONAL_ONLY, default="foo")]
)

def not_quite_too_long(x=int(f"1{'0' * (stubdefaulter.DEFAULT_LENGTH_LIMIT - 1)}"), **kwargs):
    pass
def too_long(x=int(f"1{'0' * (stubdefaulter.DEFAULT_LENGTH_LIMIT)}"), **kwargs):
    pass
"""
INPUT_STUB = """
import enum
import re
import typing
import typing_extensions
from typing import overload, Literal

def f(x: int = ..., y: str = ..., z: bool = ..., a: Any = ...) -> None: ...
def more_ints(x: int = ..., y: bool = ...) -> None: ...
def ints_as_hexadecimals(x: int = 0x7FFFFFFF, y=0b1101, z=0o744) -> None: ...
def wrong_default(wrong: int = 1) -> None: ...
def floats(a: float = ..., b: float = ..., c: float = ..., d: float = ...) -> None: ...
def float_edge_cases(one: float = ..., two: float = ..., three: float = ...) -> None: ...
def bytes_func(one: bytes = ...) -> None: ...
def containers(
    a: tuple[str, ...] = ...,
    b: list[str] = ...,
    c: dict[str, int] = ...,
    d: tuple[object, ...] = ...,
    e: list[object] = ...,
    f: set[object] = ...,
    g: dict[object, object] = ...,
) -> None: ...
def bad_container(x: set[int] = ...) -> None: ...

class Capybara:
    def __init__(self, x: int = ..., y: str = ..., z: bool = ..., a: Any = ...) -> None: ...
    @overload
    def overloaded_method(x: Literal[False] = ...) -> str: ...
    @overload
    def overloaded_method(x: Literal[True]) -> int: ...
    def __mangled(x: str = ...) -> None: ...

class Slotty: ...

class Klass:
    class NestedKlass1:
        class NestedKlass2:
            def method(self, a: bool = ...) -> None: ...
            async def async_method(self, b: float = ...) -> None: ...
    class __Mangled1:
        class __Mangled2:
            def __mangled(self, x: bool = ...) -> None: ...

@overload
def overloaded(x: Literal[False] = ...) -> str: ...
@overload
def overloaded(x: Literal[True]) -> int: ...

def intenum_default(x: int = ...) -> int: ...

class FooEnum(str, enum.Enum):
    FOO: str

def strenum_default(x: str = ...) -> str: ...
def pos_only(__x: int = ...) -> None: ...
def useless_runtime(
    *args,
    foo: None = ...,
    spam: Literal[0] = ...,
    eggs: typing.Literal["foo"] = ...,
    ham: typing_extensions.Literal[True] = ...,
    baz: Literal[1, 2, 3] = ...,
    enum_default: Literal[re.ASCII] = ...,
    **kwargs
) -> None: ...
def incorrect_non_posonly_parameter_names_in_stub(__fooooooo: str = ..., baaaaaaar: str = ..., *, baaz: str = ...) -> None: ...
def incorrect_posonly_parameter_names_in_sub(__foooooo: str = ...) -> None: ...

def not_quite_too_long(
    x: int = ...,
    *,
    y: Literal[10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000] = ...
) -> None: ...
def too_long(
    x: int = ...,
    *,
    y: Literal[100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000] = ...
) -> None: ...
"""
EXPECTED_STUB = """
import enum
import re
import typing
import typing_extensions
from typing import overload, Literal

def f(x: int = 0, y: str = 'y', z: bool = True, a: Any = None) -> None: ...
def more_ints(x: int = -1, y: bool = ...) -> None: ...
def ints_as_hexadecimals(x: int = 0x7FFFFFFF, y=0b1101, z=0o744) -> None: ...
def wrong_default(wrong: int = 1) -> None: ...
def floats(a: float = 1.23456, b: float = 0.0, c: float = -9.87654, d: float = -0.0) -> None: ...
def float_edge_cases(one: float = ..., two: float = ..., three: float = ...) -> None: ...
def bytes_func(one: bytes = b'foo') -> None: ...
def containers(
    a: tuple[str, ...] = (),
    b: list[str] = [],
    c: dict[str, int] = {},
    d: tuple[object, ...] = (1, 'foo', b'bar', True, None, 1.23),
    e: list[object] = [1, 'foo', b'bar', True, None, 1.23],
    f: set[object] = {'foo', 1, 1.23, False, None, b'bar'},
    g: dict[object, object] = {-1: 1, 'foo': 'foo', b'bar': 'bar', True: False, None: None, 1.23: 1.234},
) -> None: ...
def bad_container(x: set[int] = ...) -> None: ...

class Capybara:
    def __init__(self, x: int = 0, y: str = 'y', z: bool = True, a: Any = None) -> None: ...
    @overload
    def overloaded_method(x: Literal[False] = False) -> str: ...
    @overload
    def overloaded_method(x: Literal[True]) -> int: ...
    def __mangled(x: str = 'foo') -> None: ...

class Slotty:
    __slots__ = ('x', 'y')

class Klass:
    class NestedKlass1:
        class NestedKlass2:
            def method(self, a: bool = False) -> None: ...
            async def async_method(self, b: float = 3.14) -> None: ...
    class __Mangled1:
        class __Mangled2:
            def __mangled(self, x: bool = True) -> None: ...

@overload
def overloaded(x: Literal[False] = False) -> str: ...
@overload
def overloaded(x: Literal[True]) -> int: ...

def intenum_default(x: int = ...) -> int: ...

class FooEnum(str, enum.Enum):
    FOO: str

def strenum_default(x: str = ...) -> str: ...
def pos_only(__x: int = 5) -> None: ...
def useless_runtime(
    *args,
    foo: None = None,
    spam: Literal[0] = 0,
    eggs: typing.Literal["foo"] = "foo",
    ham: typing_extensions.Literal[True] = True,
    baz: Literal[1, 2, 3] = ...,
    enum_default: Literal[re.ASCII] = ...,
    **kwargs
) -> None: ...
def incorrect_non_posonly_parameter_names_in_stub(__fooooooo: str = ..., baaaaaaar: str = ..., *, baaz: str = ...) -> None: ...
def incorrect_posonly_parameter_names_in_sub(__foooooo: str = 'foo') -> None: ...

def not_quite_too_long(
    x: int = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000,
    *,
    y: Literal[10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000] = 10000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
) -> None: ...
def too_long(
    x: int = ...,
    *,
    y: Literal[100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000] = ...
) -> None: ...
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

        errors, _, _ = stubdefaulter.add_defaults_to_stub(
            PKG_NAME,
            typeshed_client.finder.get_search_context(search_path=[td]),
            frozenset(),
            slots=True,
        )
        assert stub_path.read_text() == EXPECTED_STUB
        assert len(errors) == 1

        stub_path.write_text(INPUT_STUB.replace(" = 1", " = ..."))
        errors, _, _ = stubdefaulter.add_defaults_to_stub(
            PKG_NAME,
            typeshed_client.finder.get_search_context(search_path=[td]),
            frozenset(),
            slots=True,
        )
        assert stub_path.read_text() == EXPECTED_STUB.replace(
            "wrong: int = 1", "wrong: int = 0"
        )
        assert len(errors) == 0


@pytest.mark.parametrize(
    "obj",
    [
        -1,
        0,
        1,
        2,
        -1.1,
        -0.0,
        0.0,
        1.1,
        True,
        False,
        None,
        "foo",
        b"bar",
        [],
        (),
        {},
        (1, "foo", b"bar", True, None, 1.23, ["foo", ("bar", 1)]),
        [1, "foo", b"bar", True, None, 1.23, (1, {b"bar": False})],
        {1},
        {-1: 1, "foo": "foo", (b"bar", b"baz"): "bar", False: [1, 2, {2, 3, 4}]},
    ],
)
def test_infer_value_of_node_known_types(obj: object) -> None:
    node = libcst.parse_expression(repr(obj))
    inferred_value = stubdefaulter.infer_value_of_node(node)
    assert inferred_value == obj
    assert type(inferred_value) is type(obj)
    assert repr(inferred_value) == repr(obj)


@pytest.mark.parametrize("obj", [3j, bytearray(), set()])
def test_infer_value_of_node_unknown_types(obj: object) -> None:
    node = libcst.parse_expression(repr(obj))
    inferred_value = stubdefaulter.infer_value_of_node(node)
    assert inferred_value is NotImplemented
