#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from pathlib import Path

import pandas as pd
import pytest

from xenonpy.descriptor import ECFP
from xenonpy.inverse.iqspr import BayesianRidgeEstimator, NGram, IQSPR


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    pwd = Path(__file__).parent
    pg_data = pd.read_csv(str(pwd / 'polymer_test_data.csv'))

    X = pg_data['smiles']
    y = pg_data.drop(['smiles', 'Unnamed: 0'], axis=1)
    ecfp = ECFP(n_jobs=1, input_type='smiles')
    bre = BayesianRidgeEstimator(descriptor=ecfp)
    ngram = NGram()
    iqspr = IQSPR(estimator=bre, modifier=ngram)
    # prepare test data
    yield dict(ecfp=ecfp, bre=bre, ngram=ngram, iqspr=iqspr, pg=(X, y))

    print('test over')


def test_base_bayesian_ridge_1(data):
    bre = data['bre']
    X, y = data['pg']
    bre.fit(X, y)

    assert 'bandgap' in bre._mdl
    assert 'refractive_index' in bre._mdl
    assert 'density' in bre._mdl
    assert 'glass_transition_temperature' in bre._mdl
    assert len(bre._mdl.keys()) == 4

    ll = bre.log_likelihood(X.sample(10),
                            bandgap=(7, 8),
                            glass_transition_temperature=(300, 400))
    assert len(ll) == 10


if __name__ == "__main__":
    pytest.main()
