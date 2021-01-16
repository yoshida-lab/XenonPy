# XenonPy contrib

Any code in this directory is not officially supported and may change or be removed at any time without notice.

The contrib directory contains project/source code directories from contributors.
These contributions/features may eventually be merged into the XenonPy package.
XenonPy is meant to be a community based project.
We encourage users to share their codes with everyone to help enhancing the functionality of XenonPy. 
While these codes may still undergo changes, updates, or extra testings,
users may create a new folder for their contributions if they are relatively complete.
If the contribution is still in the form of code fragments,
we still encourage users to share them inside the `sample_codes` folder,
where code contributions are not expected to be complete.

When adding a project, please stick to the following directory structure:

1. Create a project directory in `contrib/` with your project name, e.g., `contrib/my_project/`.
2. Provide a `README.md` under the root of the project directory, e.g., `contrib/my_project/README.md`.
3. Add unit test code (using pytest) under the `$ROOT/tests/contrib/my_project/`.

For example, if you create a project named `foo` with source file `foo.py` and the testing file `foo_test.py`. 
In `contrib/`, they are part of the project `foo`, and their full path is `$ROOT/xenonpy/contrib/foo/foo.py`,
and in `tests`, the full path is `$ROOT/tests/contrib/foo/test_foo.py`.

We prepared the `sample_codes` folder for contributors who want to share codes,
which are not complete as a stand-alone project or test codes are not available.
