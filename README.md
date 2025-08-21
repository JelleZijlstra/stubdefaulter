# stubdefaulter

A tool for automatically adding default values to Python type stubs.

## Background

Stub files, as specified in [PEP 484](https://peps.python.org/pep-0484/#stub-files)
provide a way for type checkers to see the interface to a module that is not
itself typed. Historically, default values for arguments were usually omitted
in stub files, because they are not needed for type checking. However, stub
files are also useful for IDEs, which do want to show defaults. Therefore, the
typeshed project has [decided](https://github.com/python/typeshed/issues/8988) to
allow defaults for stubs. This tool provides a way to auto-add defaults for stubs.

## Usage

Warning: The tool will import and/or install various packages. Make sure you
trust the package you are trying to add defaults for. Please run it inside a
virtual environment.

- Install the package by running `pip install stubdefaulter`
- Invoke it as `python -m stubdefaulter`
- The tool is now a linter with error codes:
  - `missing-default` for parameters that are missing a default value
  - `wrong-default` for parameters whose stub default differs from runtime
  - `missing-slots` for classes that define `__slots__` at runtime but not in the stub
- Error codes can be disabled on the command line with `--disable CODE`
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
