# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import time
from datetime import timedelta
from pathlib import Path


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
