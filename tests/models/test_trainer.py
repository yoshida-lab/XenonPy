#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import glob
import os

import pytest

from xenonpy.model.training.base import BaseRunner


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


def test_base_runner_1(data):
    assert BaseRunner.check_cuda(False).type == 'cpu'
    assert BaseRunner.check_cuda('cpu').type == 'cpu'

    with pytest.raises(RuntimeError, match='could not use CUDA on this machine'):
        BaseRunner.check_cuda(True)

    with pytest.raises(RuntimeError, match='could not use CUDA on this machine'):
        BaseRunner.check_cuda('cuda')

    with pytest.raises(RuntimeError, match='wrong device identifier'):
        BaseRunner.check_cuda('other illegal')


if __name__ == "__main__":
    pytest.main()
