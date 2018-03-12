# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import numpy as np
from numpy import product


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


def absolute_path(path, ignore_err=True):
    """
    Resolve path when path include ``~``, ``parent/here``.

    Parameters
    ----------
    path: str
        Path to expand.
    ignore_err: bool
        FileNotFoundError is raised when set to False.
        When True, the path will be created.
    Returns
    -------
    str
        Expanded path.
    """
    from sys import version_info

    p = Path(path)
    if version_info[1] == 5:
        if ignore_err:
            p.expanduser().mkdir(parents=True, exist_ok=True)
        return str(p.expanduser().resolve())
    return str(p.expanduser().resolve(not ignore_err))


class Stopwatch(object):
    def __init__(self):
        self._start = time.monotonic()

    @property
    def count(self):
        return timedelta(seconds=time.monotonic() - self._start)


class Product(object):
    def __init__(self, *paras, repeat=1):
        if not isinstance(repeat, int):
            raise ValueError('repeat must be int but got {}'.format(type(repeat)))
        lens = [len(p) for p in paras]
        if repeat > 1:
            lens = lens * repeat
        size = product(lens)
        acc_list = [np.floor_divide(size, lens[0])]
        for len_ in lens[1:]:
            acc_list.append(np.floor_divide(acc_list[-1], len_))

        self.paras = paras * repeat if repeat > 1 else paras
        self.lens = lens
        self.size = size
        self.acc_list = acc_list

    def __getitem__(self, index):
        if index > self.size - 1:
            raise IndexError
        ret = [s - 1 for s in self.lens]  # from len to index
        remainder = index + 1
        for i, acc in enumerate(self.acc_list):
            quotient, remainder = np.divmod(remainder, acc)
            if remainder == 0:
                ret[i] = quotient - 1
                break
            ret[i] = quotient

        return tuple(self.paras[i][j] for i, j in enumerate(ret))

    def __len__(self):
        return self.size


def get_sha256(fname):
    """
    Calculate file's sha256 value

    Parameters
    ----------
    fname: str
        File name.

    Returns
    -------
    str
        sha256 value.
    """
    from hashlib import sha256
    hasher = sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
