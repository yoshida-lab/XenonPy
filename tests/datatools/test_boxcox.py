#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pytest
from sklearn.preprocessing import PowerTransformer as PT

from xenonpy.datatools.transform import PowerTransformer


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

    # raw_shift = raw - raw.min() + 1e-9
    a_ = PT(standardize=False).fit_transform(raw.reshape(-1, 1))
    a_ = a_.reshape(-1, 1)
    trans_4x1 = a_
    trans_4x4 = np.concatenate((a_, a_, a_, a_), axis=1)

    raw_err = np.array([1., 1., 1., 1.])
    a = raw_err.reshape(-1, 1)
    raw_err_4x1 = a
    raw_err_4x4 = np.concatenate((a, a, a, a), axis=1)

    a_ = np.log(raw_err)
    a_ = a_.reshape(-1, 1)
    trans_err_4x1 = a_
    trans_err_4x4 = np.concatenate((a_, a_, a_, a_), axis=1)
    yield raw_4x1, raw_4x4, trans_4x1, trans_4x4, raw_err_4x1, raw_err_4x4, trans_err_4x1, trans_err_4x4

    print('test over')


def test_transform_4x1_1(data):
    bc = PowerTransformer()
    trans = bc.fit_transform(data[0])
    assert np.all(trans == data[2])
    assert trans.shape == data[2].shape
    inverse = bc.inverse_transform(trans)
    assert np.allclose(inverse, data[0])
    assert inverse.shape == data[0].shape


def test_transform_4x1_3(data):
    bc = PowerTransformer()
    bc.fit_transform(data[0])
    trans = bc.transform(data[0][:2])
    assert trans.shape == data[2][:2].shape
    assert np.all(trans == data[2][:2])
    inverse = bc.inverse_transform(trans)
    assert inverse.shape == data[0][:2].shape
    assert np.allclose(inverse, data[0][:2])


def test_transform_4x1_ravel_1(data):
    bc = PowerTransformer()
    trans = bc.fit_transform(data[0].ravel())
    print(trans)
    print(data[2].ravel())
    assert np.all(trans == data[2].ravel())
    inverse = bc.inverse_transform(trans)
    assert np.allclose(inverse, data[0].ravel())


def test_transform_4x1_ravel_2(data):
    bc = PowerTransformer()
    bc.fit(data[0].ravel())
    trans = bc.transform((data[0].ravel())[:2])
    assert trans.shape == (data[2].ravel())[:2].shape
    assert np.all(trans == (data[2].ravel())[:2])
    inverse = bc.inverse_transform(trans)
    assert inverse.shape == (data[0].ravel())[:2].shape
    assert np.allclose(inverse, (data[0].ravel())[:2])


def test_transform_4x4_1(data):
    bc = PowerTransformer()
    bc.fit_transform(data[1])
    trans = bc.transform(data[1][:2])
    assert trans.shape == data[3][:2].shape
    assert np.all(trans == data[3][:2])
    inverse = bc.inverse_transform(trans)
    assert inverse.shape == data[1][:2].shape
    assert np.allclose(inverse, data[1][:2])


def test_transform_4x4_2(data):
    bc = PowerTransformer()
    trans = bc.fit_transform(data[1])
    assert np.all(trans == data[3])
    inverse = bc.inverse_transform(trans)
    assert np.allclose(inverse, data[1])


def test_transform_err_4x1(data):
    bc = PowerTransformer()
    trans = bc.fit_transform(data[4])
    assert np.all(trans == data[4])
    inverse = bc.inverse_transform(trans)
    assert np.all(inverse == data[4])


def test_transform_err_4x4(data):
    bc = PowerTransformer()
    trans = bc.fit_transform(data[5])
    assert np.all(trans == data[5])
    inverse = bc.inverse_transform(trans)
    assert np.all(inverse == data[5])


def test_transform_err_4x1_2(data):
    bc = PowerTransformer(on_err='nan')
    trans = bc.fit_transform(data[4])
    assert np.all(np.isnan(trans))


def test_transform_err_4x4_2(data):
    bc = PowerTransformer(on_err='nan')
    trans = bc.fit_transform(data[5])
    assert np.all(np.isnan(trans))


def test_transform_err_4x1_3(data):
    bc = PowerTransformer(on_err='log')
    trans = bc.fit_transform(data[4])
    assert np.all(trans == data[6])
    inverse = bc.inverse_transform(trans)
    assert np.all(inverse == data[4])


def test_transform_err_4x4_3(data):
    bc = PowerTransformer(on_err='log')
    trans = bc.fit_transform(data[5])
    assert np.all(trans == data[7])
    inverse = bc.inverse_transform(trans)
    assert np.all(inverse == data[5])


def test_transform_err():
    with pytest.raises(ValueError):
        bc = PowerTransformer(on_err='raise')
        bc.fit_transform([1, 1, 1])
    with pytest.raises(FloatingPointError):
        bc = PowerTransformer(on_err='raise')
        bc.fit_transform([1e20, 1, 1e20])


if __name__ == "__main__":
    pytest.main()
