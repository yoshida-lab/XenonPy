#  Copyright 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pytest

from xenonpy.descriptor import ECFP
from xenonpy.inverse.iqspr import BayesianRidgeEstimator


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    desc = ECFP(n_jobs=1)
    # prepare test data
    yield dict(desc=desc)

    print('test over')


def test_base_bayesian_ridge(data):
    br = BayesianRidgeEstimator(descriptor=data['desc'])
    
    X = np.array([1, 2, 3, 3, 4, 4])
    ll = llh(X, 10)
    assert np.all(ll == np.array([1, 2, 3, 3, 4, 4]) * 10)


if __name__ == "__main__":
    pytest.main()
