# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from contextlib import contextmanager
from os import getcwd
from pathlib import Path


@contextmanager
def set_env(**kwargs):
    """
    Set temp environment variable with ``with`` statement.

    Examples
    --------
    >>> import os
    >>> with set_env(test='test env'):
    >>>    print(os.getenv('test'))
    test env
    >>> print(os.getenv('test'))
    None

    Parameters
    ----------
    kwargs: dict
        Dict with string value.
    """
    import os

    tmp = dict()
    for k, v in kwargs.items():
        tmp[k] = os.getenv(k)
        os.environ[k] = v
    yield
    for k, v in tmp.items():
        if not v:
            del os.environ[k]
        else:
            os.environ[k] = v


def expand_path(path):
    """
    Expand path when path include ``~``, ``parent/here``.

    Parameters
    ----------
    path: str
        Path to expand.
    Returns
    -------
    str
        Expanded path.
    """
    from platform import system

    if system() == 'Windows':
        if path[1] != ':':
            return str(Path(getcwd() + '/' + str(Path(path).expanduser())))
    else:
        if path[0] != '/':
            return getcwd() + '/' + str(Path(path).expanduser())
    return str(Path(path).expanduser())
