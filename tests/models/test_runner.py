# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import glob
import os

import pytest

from xenonpy.model.nn import BaseRunner


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield
    r = glob.glob('log_*.txt')
    for i in r:
        os.remove(i)
    print('test over')


def test_base_runner(data):
    with BaseRunner() as runner:
        assert hasattr(runner, '__enter__')
        assert hasattr(runner, '__exit__')


if __name__ == "__main__":
    pytest.main()
