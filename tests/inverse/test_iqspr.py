#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import BayesianRidge

from xenonpy.descriptor import ECFP
from xenonpy.inverse.iqspr import BayesianRidgeEstimator, NGram, IQSPR, GetProbError


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
    models = bre.estimators

    assert 'bandgap' in bre._mdl
    assert 'refractive_index' in bre._mdl
    assert 'density' in bre._mdl
    assert 'glass_transition_temperature' in bre._mdl

    assert 'bandgap' in models
    assert 'refractive_index' in models
    assert 'density' in models
    assert 'glass_transition_temperature' in models

    assert len(models.keys()) == 4

    ll = bre.log_likelihood(X.sample(10),
                            bandgap=(7, 8),
                            glass_transition_temperature=(300, 400))
    assert len(ll) == 10
    assert isinstance(bre['bandgap'], BayesianRidge)
    assert isinstance(bre['density'], BayesianRidge)

    with pytest.raises(KeyError):
        bre['other']

    with pytest.raises(TypeError):
        bre['other'] = 1
    bre['other'] = BayesianRidge()


def test_ngram_1(data):
    ngram = data['ngram']
    assert ngram.ngram_table is None
    assert ngram.max_len == 1000
    assert ngram.del_range == (1, 10)
    assert ngram.reorder_prob == 0
    assert ngram.sample_order == 10
    assert ngram._train_order is None

    ngram.set_params(max_len=500, reorder_prob=0.2)

    assert ngram.max_len == 500
    assert ngram.del_range == (1, 10)
    assert ngram.reorder_prob == 0.2

    def on_errors(self, error, smi):
        raise error

    ngram.fit(data['pg'][0][:20], train_order=5)

    assert ngram._train_order == 5
    assert ngram.sample_order == 5
    assert ngram.ngram_table is not None

    np.random.seed(123456)
    ngram.proposal(data['pg'][0][60:65])

    ngram.on_errors = types.MethodType(on_errors, ngram)
    np.random.seed(123456)
    with pytest.raises(GetProbError):
        ngram.proposal(data['pg'][0][60:65])


def test_iqspr_1(data):
    np.random.seed(0)
    ecfp = ECFP(n_jobs=1, input_type='smiles')
    bre = BayesianRidgeEstimator(descriptor=ecfp)
    ngram = NGram()
    iqspr = IQSPR(estimator=bre, modifier=ngram)
    X, y = data['pg']
    bre.fit(X, y)
    ngram.fit(data['pg'][0][0:20], train_order=10)
    beta = np.linspace(0.05, 1, 10)
    for s, ll, p, f in iqspr(data['pg'][0][:5], beta, yield_lpf=True, bandgap=(0.1, 0.2), density=(0.9, 1.2)):
        assert np.abs(np.sum(p) - 1.0) < 1e-5
        assert np.sum(f) == 5, print(f)


if __name__ == "__main__":
    pytest.main()
