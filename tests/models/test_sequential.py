#  Copyright (c) 2021. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pytest
import torch

from xenonpy.model import LinearLayer, SequentialLinear


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    x = torch.randn(10, 10)

    yield (x,)
    print('test over')


def test_layer_1(data):
    layer = LinearLayer(10, 1)
    assert layer.linear.in_features == 10
    assert layer.linear.out_features == 1
    assert layer.linear.bias is not None
    assert layer.dropout.p == 0
    assert layer.activation.__class__.__name__ == 'ReLU'
    assert layer.normalizer.__class__.__name__ == 'BatchNorm1d'

    x = layer(data[0])
    assert x.size() == (10, 1)


def test_layer_2(data):
    layer = LinearLayer(10, 5, bias=False, dropout=0.5, normalizer=None, activation_func=torch.nn.Tanh())
    assert layer.linear.in_features == 10
    assert layer.linear.out_features == 5
    assert layer.linear.bias is None
    assert layer.dropout.p == 0.5
    assert layer.activation.__class__.__name__ == 'Tanh'
    assert layer.normalizer is None

    x = layer(data[0])
    assert x.size() == (10, 5)


def test_sequential_1(data):
    m = SequentialLinear(10, 5)
    assert isinstance(m, torch.nn.Module)
    assert m.output.in_features == 10
    assert m.output.out_features == 5

    x = m(data[0])
    assert x.size() == (10, 5)


def test_sequential_2(data):
    m = SequentialLinear(10, 1, bias=False, h_neurons=(7, 4))
    assert isinstance(m.layer_0, LinearLayer)
    assert isinstance(m.layer_1, LinearLayer)
    assert m.layer_0.linear.in_features == 10
    assert m.layer_0.linear.out_features == 7
    assert m.layer_0.linear.bias is None
    assert m.layer_1.linear.in_features == 7
    assert m.layer_1.linear.out_features == 4
    assert m.layer_1.linear.bias is not None
    assert m.output.in_features == 4
    assert m.output.out_features == 1
    assert m.output.bias is not None

    x = m(data[0])
    assert x.size() == (10, 1)


def test_sequential_3():
    m = SequentialLinear(10, 1, bias=False, h_neurons=(0.7, 0.5))
    assert m.layer_0.linear.in_features == 10
    assert m.layer_0.linear.out_features == 7
    assert m.layer_0.linear.bias is None
    assert m.layer_1.linear.in_features == 7
    assert m.layer_1.linear.out_features == 5
    assert m.layer_1.linear.bias is not None
    assert m.output.in_features == 5
    assert m.output.out_features == 1
    assert m.output.bias is not None


def test_sequential_4():
    with pytest.raises(RuntimeError, match='illegal parameter type of <h_neurons>'):
        SequentialLinear(10, 5, h_neurons=('NaN',))

    with pytest.raises(RuntimeError, match='number of parameter not consistent with number of layers'):
        SequentialLinear(10, 5, h_neurons=(4,), h_dropouts=(1, 5))

    m = SequentialLinear(10, 1, h_neurons=(3, 4))
    tmp = (1, 2)
    assert id(m._check_input(tmp)) == id(tmp)


if __name__ == "__main__":
    pytest.main()
