# XenonPy contrib

Any code in this directory is not officially supported and may change or be
removed at any time without notice.

The contrib directory contains project/source code directories from contributors.
It is meant to contain features and contributions that eventually should
get merged into XenonPy, but whose interfaces may still change, or which
require some testing to see whether they can find broader acceptance.

When adding a project, please stick to the following directory structure:
1. Create a project directory in `contrib/`, and mirror the portions of the XenonPy tree that your project requires underneath `contrib/my_project/`.
2. Provide a `README.md` under the root of the project directory, e.g  `contrib/my_project/README.md`.


For example, let's say you create a project named `foo` with source file `foo.py` and the testing file
`foo_test.py`. If you were to merge those files directly into XenonPy,
they would live in `$ROOT/xenonpy/descriptor/foo.py` and
`$ROOT/tests/descriptor/foo_test.py`. In `contrib/`, they are part
of project `foo`, and their full paths are `$ROOT/xenonpy/contrib/foo/descriptor/foo.py`
and in `tests`, the full paths are `$ROOT/tests/foo/descriptor/test_foo.py`.
