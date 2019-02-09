#  Copyright 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pytest
import torch as tc

from xenonpy.model.nn import Generator1d, Layer1d
from xenonpy.model.nn.wrap import L1


def test_layer():
    layer = Layer1d(10, 1)
    assert isinstance(layer, tc.nn.Module)


def test_generator1d1():
    g = Generator1d(290, 1, n_neuron=(100, 70, 50), drop_out=(0.2, 0.3, 0.4),
                    batch_normalize=(None, L1.batch_norm()))
    m = g(1)
    assert len(list(m)) == 18, '3x3x2'


def test_generator1d2():
    g = Generator1d(290, 1, n_neuron=[100, 70, 50], drop_out=(0.2, 0.3, 0.4),
                    batch_normalize=(None, L1.batch_norm()))
    m = g(1, n_models=10)
    assert len(list(m)) == 10, '0 < n_models <= 3x3x2'


def test_generator1d3():
    g = Generator1d(290, 1, n_neuron=[100, 70, 50], drop_out=(0.2, 0.3, 0.4),
                    batch_normalize=(None, L1.batch_norm()))
    m = g(1, n_models=20)
    try:
        len(list(m)) == 18
    except ValueError:
        assert True, "n_models > 3x3x2, larger sample than population when 'replace=False'"
    else:
        assert False, 'should got ValueError:'


def test_generator1d4():
    g = Generator1d(290, 1, n_neuron=[100, 90, 80, 70, 60, 50], drop_out=(0.2, 0.3, 0.4),
                    batch_normalize=(None, L1.batch_norm()))
    m = g(5, n_models=20, replace=True)
    try:
        assert len(list(m)) == 20, "when 'replace=False' is OK"
    except ValueError:
        assert False, "should no Error"


if __name__ == "__main__":
    pytest.main()
