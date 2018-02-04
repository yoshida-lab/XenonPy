# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from .datatools import Loader, Saver

from contextlib import contextmanager


@contextmanager
def set_env(**kwargs):
    """
    Set temp environment variable.
    This is a ``with`` statement.

    Parameters
    ----------
    kwargs: dict
        String dict with ``env=val``
    """
    import os

    tmp = dict()
    for k, v in kwargs.items():
        tmp[k] = os.getenv(k)
        os.environ[k] = v
    yield
    for k, v in tmp.items():
        os.environ[k] = v
