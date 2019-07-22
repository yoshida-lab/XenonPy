#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import pytest
import torch

from xenonpy.model.nn import check_cuda, to_tensor


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield
    # r = glob.glob('log_*.txt')
    # for i in r:
    #     os.remove(i)
    print('test over')


def test_check_cuda():
    assert check_cuda(False).type == 'cpu'
    assert check_cuda('cpu').type == 'cpu'

    with pytest.raises(RuntimeError, match='could not use CUDA on this machine'):
        check_cuda(True)

    with pytest.raises(RuntimeError, match='could not use CUDA on this machine'):
        check_cuda('cuda')

    with pytest.raises(RuntimeError, match='wrong device identifier'):
        check_cuda('other illegal')


def test_to_tensor():
    ls = [[1, 2, 3], [4, 5, 6]]
    tp = tuple(ls)
    np_ = np.asarray(tp)
    pd_ = pd.DataFrame(ls)
    tensor_ = torch.Tensor(ls)

    t = to_tensor(ls)
    assert t.shape == (2, 3)
    assert isinstance(t, torch.Tensor)
    t = to_tensor(ls, unsqueeze=-1)
    assert t.shape == (2, 3, 1)

    t = to_tensor(tp)
    assert t.shape == (2, 3)
    assert isinstance(t, torch.Tensor)
    t = to_tensor(tp, unsqueeze=-1)
    assert t.shape == (2, 3, 1)

    t = to_tensor(np_)
    assert t.shape == (2, 3)
    assert isinstance(t, torch.Tensor)
    t = to_tensor(np_, unsqueeze=-1)
    assert t.shape == (2, 3, 1)

    t = to_tensor(pd_)
    assert t.shape == (2, 3)
    assert isinstance(t, torch.Tensor)
    t = to_tensor(pd_, unsqueeze=-1)
    assert t.shape == (2, 3, 1)

    t = to_tensor(tensor_)
    assert t.shape == (2, 3)
    assert isinstance(t, torch.Tensor)
    t = to_tensor(tensor_, unsqueeze=-1)
    assert t.shape == (2, 3, 1)

    with pytest.raises(RuntimeError,
                       match='input must be pd.DataFrame, pd.Series, np.ndarray, list, tuple, or torch.Tensor'):
        to_tensor('illegal params')


if __name__ == "__main__":
    pytest.main()
