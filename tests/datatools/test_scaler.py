#  Copyright 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import numpy as np
import pytest
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler

from xenonpy.datatools.transform import Scaler


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
    a_, _ = boxcox(raw)
    a_ = a_.reshape(-1, 1)
    trans_4x1 = a_
    trans_4x4 = np.concatenate((a_, a_, a_, a_), axis=1)

    raw_err = np.array([1., 1., 1., 1.])
    a = raw_err.reshape(-1, 1)
    raw_err_4x1 = a
    raw_err_4x4 = np.concatenate((a, a, a, a), axis=1)

    a_ = boxcox(raw_err, 0)
    a_ = a_.reshape(-1, 1)
    trans_err_4x1 = a_
    trans_err_4x4 = np.concatenate((a_, a_, a_, a_), axis=1)
    yield raw_4x1, raw_4x4, trans_4x1, trans_4x4, raw_err_4x1, raw_err_4x4, trans_err_4x1, trans_err_4x4

    print('test over')


def test_scaler_1(data):
    bc = Scaler().box_cox()
    std = Scaler().standard()
    bc_std = Scaler().box_cox().standard()
    bc_trans = bc.fit_transform(data[0])
    std_trans = std.fit_transform(bc_trans)
    bc_std_trans = bc_std.fit_transform(data[0])
    assert np.allclose(bc_trans, data[2])
    assert np.allclose(std_trans, StandardScaler().fit_transform(bc_trans))
    assert np.allclose(bc_std_trans, std_trans)
    inverse = bc_std.inverse_transform(bc_std_trans)
    assert np.allclose(inverse, data[0])


if __name__ == "__main__":
    pytest.main()
