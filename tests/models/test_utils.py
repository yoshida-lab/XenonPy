#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pytest

from xenonpy.model.utils import regression_metrics


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    noise = 0.001
    x = np.random.randn(100)
    y = x + np.random.rand() * noise

    yield x, y, noise
    print('test over')


def test_regression_metrics_1(data):
    x = data[0]
    y = data[1]
    noise = data[2]
    metric = regression_metrics(x, y)
    assert isinstance(metric, dict)
    assert set(metric.keys()) == {'mae', 'mse', 'rmse', 'r2', 'pearsonr', 'spearmanr', 'p_value', 'max_ae'}
    assert metric['mae'] < noise
    assert metric['mse'] < noise ** 2
    assert metric['rmse'] < noise
    assert metric['r2'] > 0.9999
    assert metric['pearsonr'] > 0.9999
    assert metric['spearmanr'] > 0.9999
    assert np.isclose(metric['p_value'], 0, 1e-4)
    assert metric['max_ae'] < noise

    assert metric['mae'] == regression_metrics(x.reshape(-1, 1), y)['mae']
    assert metric['rmse'] == regression_metrics(x.reshape(-1, 1), y.reshape(-1, 1))['rmse']
    assert metric['max_ae'] == regression_metrics(x, y.reshape(-1, 1))['max_ae']


if __name__ == "__main__":
    pytest.main()
