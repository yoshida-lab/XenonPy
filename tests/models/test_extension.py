#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import torch

from xenonpy.model.training.base import BaseExtension, BaseRunner
from xenonpy.model.training.extension import TensorConverter, Validator
from xenonpy.model.utils import regression_metrics


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    x = np.random.randn(100)
    y = x + np.random.rand() * 0.001

    yield x, y
    print('test over')


def test_base_runner_1():
    ext = BaseExtension()
    x, y = 1, 2
    assert ext.input_proc(x, y) == (x, y)
    assert ext.output_proc(y) == (y, None)

    x, y = (1,), 2
    assert ext.input_proc(x, y) == (x, y)
    assert ext.output_proc(y) == (y, None)

    x, y = (1,), (2,)
    assert ext.input_proc(x, y) == (x, y)
    assert ext.output_proc(y, y) == (y, y)


def test_tensor_converter_1():
    class _Trainer(BaseRunner):
        def __init__(self):
            super().__init__()

        def predict(self, x_, y_):
            return x_, y_

    trainer = _Trainer()
    converter = TensorConverter()
    np_ = np.asarray([[1, 2, 3], [4, 5, 6]])
    pd_ = pd.DataFrame(np_)
    tensor_ = torch.Tensor(np_)

    x, y = converter.input_proc(np_, None, trainer=trainer)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert y is None

    x, y = converter.input_proc(pd_, None, trainer=trainer)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert y is None

    x, y = converter.input_proc(tensor_, None, trainer=trainer)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert y is None

    x, y = converter.input_proc(np_, np_, trainer=trainer)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert torch.equal(y, tensor_)

    x, y = converter.input_proc(pd_, pd_, trainer=trainer)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert torch.equal(y, tensor_)

    x, y = converter.input_proc(tensor_, tensor_, trainer=trainer)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (2, 3)
    assert torch.equal(x, tensor_)
    assert torch.equal(y, tensor_)


def test_tensor_converter_2():
    class _Trainer(BaseRunner):
        def __init__(self):
            super().__init__()

        def predict(self, x_, y_):
            return x_, y_

    trainer = _Trainer()
    converter = TensorConverter()
    np_ = np.asarray([[1, 2, 3], [4, 5, 6]])
    pd_ = pd.DataFrame(np_)
    tensor_ = torch.Tensor(np_)

    x, y = converter.input_proc(np_, np_[0], trainer=trainer)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 1)
    assert torch.equal(y, tensor_[0].unsqueeze(-1))

    x, y = converter.input_proc(pd_, pd_.iloc[0], trainer=trainer)
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3, 1)
    assert torch.equal(y, tensor_[0].unsqueeze(-1))

    x, y = converter.input_proc(tensor_, tensor_[0], trainer=trainer)
    print(tensor_[0].size())
    assert isinstance(y, torch.Tensor)
    assert y.shape == (3,)
    assert torch.equal(y, tensor_[0])


def test_tensor_converter_3():
    converter = TensorConverter()
    np_ = np.asarray([[1, 2, 3], [4, 5, 6]])
    tensor_ = torch.from_numpy(np_)

    y, y_ = converter.output_proc(tensor_, training=True)
    assert y_ is None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (2, 3)
    assert torch.equal(y, tensor_)

    y, y_ = converter.output_proc(tensor_, tensor_, training=True)
    assert isinstance(y, torch.Tensor)
    assert isinstance(y_, torch.Tensor)
    assert y.equal(y_)
    assert y.shape == (2, 3)
    assert torch.equal(y, tensor_)

    y, _ = converter.output_proc((tensor_,), training=True)
    assert isinstance(y, tuple)
    assert isinstance(y[0], torch.Tensor)
    assert torch.equal(y[0], tensor_)

    y, y_ = converter.output_proc(tensor_, tensor_, training=False)
    assert isinstance(y, np.ndarray)
    assert isinstance(y_, np.ndarray)
    assert np.all(y == y_)
    assert y.shape == (2, 3)
    assert np.all(y == tensor_.numpy())

    y, _ = converter.output_proc((tensor_,), training=False)
    assert isinstance(y, tuple)
    assert isinstance(y[0], np.ndarray)
    assert np.all(y[0] == tensor_.numpy())


def test_validator_1(data):
    x = data[0]
    y = data[1]

    class _Trainer(BaseRunner):
        def __init__(self):
            super().__init__()
            self.x_val = x
            self.y_val = y

        def predict(self, x_, y_):
            return x_, y_

    val = Validator(metrics_func=regression_metrics)

    step_info = OrderedDict()
    assert bool(step_info) is False

    val.step_forward(step_info, trainer=_Trainer())
    assert set(step_info.keys()) == {'val_mae', 'val_mse', 'val_rmse', 'val_r2', 'val_pearsonr', 'val_spearmanr',
                                     'val_p_value', 'val_max_error'}
    assert step_info['val_mae'] == regression_metrics(x, y)['mae']


if __name__ == "__main__":
    pytest.main()
