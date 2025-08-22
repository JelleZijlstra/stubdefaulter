# stubdefaulter

A tool for linting and autofixing Python type stubs.

Currently, it supports the following checks:
- `missing-default`: parameters missing a default value (has autofix)
- `wrong-default`: the default value of a parameter differs from the runtime
- `missing-slots`: for classes that define `__slots__` at runtime but not in the stub (has autofix)
- `disjoint-base-with-slots`: remove `@disjoint_base` decorator (PEP 800) for classes that
  also define `__slots__` (has autofix)

## Background

Stub files, as specified in [PEP 484](https://peps.python.org/pep-0484/#stub-files)
provide a way for type checkers to see the interface to a module that is not
itself typed. Historically, default values for arguments were usually omitted
in stub files, because they are not needed for type checking. However, stub
files are also useful for IDEs, which do want to show defaults. Therefore, the
typeshed project has [decided](https://github.com/python/typeshed/issues/8988) to
allow defaults for stubs. This tool provides a way to auto-add defaults for stubs.

More generally, this tool provides the following useful combination of features
for linting stubs:
- Runtime-aware: it can compare stubs against the corresponding runtime object
- Capable of autofixes

Therefore, I am now expanding it with other useful checks. Comparable tools include:
- `stubtest`, a tool distributed with mypy that compares stubs against the runtime
  and produces errors, but not autofixes
- `stubgen`, another tool bundled with mypy that can generate stubs from scratch
- `flake8-pyi`, a linter for stubs
- `ruff`, a general Python linter; it incorporates the rules from flake8-pyi and adds
  some autofixes
- `docstring-adder`, a tool that adds docstrings to stubs

## Usage

Warning: The tool will import and/or install various packages. Make sure you
trust the package you are trying to add defaults for. Please run it inside a
virtual environment.

- Install the package by running `pip install stubdefaulter`
- Invoke it as `python -m stubdefaulter`
- By default, all checks are enabled. Error codes can be disabled on the command line with `--disable CODE`
- Use `--fix` to apply autofixes for the selected error codes
- Example invocations:
  - `python -m stubdefaulter --fix --stdlib-path path/to/typeshed/stdlib`
    - Add defaults to the stdlib stubs in typeshed
  - `python -m stubdefaulter --packages path/to/typeshed/stubs/requests path/to/typeshed/stubs/babel`
    - Add defaults to the `requests` and `babel` packages in typeshed
    - Assumes you already have these installed locally
  - `python -m stubdefaulter --typeshed-packages path/to/typeshed/stubs/requests path/to/typeshed/stubs/babel`
    - Like the above, but also _automatically installs_ the version of the
      package that typeshed supports

## Limitations/TODOs

- Does not add values to variables and class attributes

## Changelog

### Version 0.1.0 (May 1, 2023)

Initial PyPI release.
