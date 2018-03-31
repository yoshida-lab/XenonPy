# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


import numpy as np
import pytest
from scipy.stats import boxcox

from xenonpy.preprocess import BoxCox


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    # prepare test data
    raw = np.array([1., 2., 3., 4.])
    a = raw.reshape(-1, 1)
    raw_4x1 = a
    raw_4x4 = np.concatenate((a, a, a, a), axis=1)

    raw_shift = raw - raw.min() + 1e-9
    a_, _ = boxcox(raw_shift)
    a_ = a_.reshape(-1, 1)
    trans_4x1 = a_
    trans_4x4 = np.concatenate((a_, a_, a_, a_), axis=1)

    raw_err = np.array([1., 1., 1., 1.])
    a = raw_err.reshape(-1, 1)
    raw_err_4x1 = a
    raw_err_4x4 = np.concatenate((a, a, a, a), axis=1)

    raw_err_shift = raw_err - raw_err.min() + 1e-9
    a_ = boxcox(raw_err_shift, 0)
    a_ = a_.reshape(-1, 1)
    trans_err_4x1 = a_
    trans_err_4x4 = np.concatenate((a_, a_, a_, a_), axis=1)
    yield raw_4x1, raw_4x4, trans_4x1, trans_4x4, raw_err_4x1, raw_err_4x4, trans_err_4x1, trans_err_4x4

    print('test over')


def test_transform_4x1(data):
    bc = BoxCox()
    trans = bc.fit_transform(data[0])
    assert np.all(trans == data[2])
    inverse = bc.inverse_transform(trans)
    assert np.all(inverse == data[0])


def test_transform_4x4(data):
    bc = BoxCox()
    trans = bc.fit_transform(data[1])
    assert np.all(trans == data[3])
    inverse = bc.inverse_transform(trans)
    assert np.all(inverse == data[1])


def test_transform_err_4x1(data):
    bc = BoxCox()
    trans = bc.fit_transform(data[4])
    assert np.all(trans == data[6])
    inverse = bc.inverse_transform(trans)
    print(inverse)
    assert np.all(inverse == data[4])


def test_transform_err_4x4(data):
    bc = BoxCox()
    trans = bc.fit_transform(data[5])
    assert np.all(trans == data[7])
    inverse = bc.inverse_transform(trans)
    assert np.all(inverse == data[5])


if __name__ == "__main__":
    pytest.main()
