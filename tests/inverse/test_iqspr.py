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
from xenonpy.inverse.iqspr import GaussianLogLikelihood, NGram, IQSPR, GetProbError, MolConvertError


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
    bre = GaussianLogLikelihood(descriptor=ecfp)
    ngram = NGram()
    iqspr = IQSPR(estimator=bre, modifier=ngram)
    # prepare test data
    yield dict(ecfp=ecfp, bre=bre, ngram=ngram, iqspr=iqspr, pg=(X, y))

    print('test over')


def test_gaussian_ll_1(data):
    bre = data['bre']
    X, y = data['pg']
    bre.fit(X, y)

    assert 'bandgap' in bre._mdl
    assert 'refractive_index' in bre._mdl
    assert 'density' in bre._mdl
    assert 'glass_transition_temperature' in bre._mdl

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

    bre.remove_estimator()
    assert bre._mdl == {}


def test_gaussian_ll_2(data):
    bre = data['bre']
    X, y = data['pg']
    bre.fit(X, y)

    x = X.sample(10)
    tmp = bre.predict(x)
    assert isinstance(tmp, pd.DataFrame)
    assert tmp.shape == (10, 8)

    x[66666] = 'CCd'
    tmp = bre.predict(x)
    assert isinstance(tmp, pd.DataFrame)
    assert tmp.shape == (11, 8)
    print(tmp)
    tmp = tmp.loc[66666]

    assert tmp.isna().all()


def test_ngram_1(data):
    ngram = NGram()
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


def test_ngram_2(data):
    ngram = NGram()

    with pytest.warns(RuntimeWarning, match='<sample_order>'):
        ngram.fit(data['pg'][0][:20], train_order=5)

    assert ngram._train_order == 5
    assert ngram.sample_order == 5
    assert ngram.ngram_table is not None

    np.random.seed(123456)
    with pytest.warns(RuntimeWarning, match='can not convert'):
        old_smis = ['CC(=S)C([*])(C)=CCC([*])']
        tmp = ngram.proposal(old_smis)
        assert tmp == old_smis

    np.random.seed(654321)
    with pytest.warns(RuntimeWarning, match='get_prob: '):
        old_smis = ['C([*])C([*])(C1=C(OCCC)C=CC(Br)C1)']
        tmp = ngram.proposal(old_smis)
        assert tmp == old_smis


def test_ngram_3(data):
    ngram = NGram(sample_order=5)
    ngram.fit(data['pg'][0][:20], train_order=5)

    def on_errors(self, error):
        if isinstance(error, MolConvertError):
            raise error
        else:
            return error.old_smi

    np.random.seed(123456)
    ngram.on_errors = types.MethodType(on_errors, ngram)
    with pytest.raises(MolConvertError):
        old_smis = ['CC(=S)C([*])(C)=CCC([*])']
        ngram.proposal(old_smis)

    def on_errors(self, error):
        if isinstance(error, GetProbError):
            raise error
        else:
            return error.old_smi

    np.random.seed(654321)
    ngram.on_errors = types.MethodType(on_errors, ngram)
    with pytest.raises(GetProbError):
        old_smis = ['C([*])C([*])(C1=C(OCCC)C=CC(Br)C1)']
        ngram.proposal(old_smis)


def test_ngram_4():
    smis1 = ['CCCc1ccccc1', 'CC(CCc1ccccc1)CC', 'Cc1ccccc1CC', 'C(CC(C))CC', 'CCCC']
    n_gram1 = NGram()  # base case
    n_gram1.fit(smis1, train_order=3)
    tmp_tab = n_gram1._table

    assert (len(tmp_tab),len(tmp_tab[0][0])) == (3,2)

    check_close = [[(4, 5), (2, 2)], [(6, 5), (3, 2)], [(8, 4), (4, 2)]]
    check_open = [[(5, 5), (2, 2)], [(6, 5), (3, 2)], [(6, 5), (4, 2)]]

    for ii, x in enumerate(tmp_tab):
        for i in range(len(x[0])):
            assert x[0][i].shape == check_close[ii][i]
            assert x[1][i].shape == check_open[ii][i]


def test_ngram_5():
    smis1 = ['CCCc1ccccc1', 'CC(CCc1ccccc1)CC', 'Cc1ccccc1CC', 'C(CC(C))CC', 'CCCC']
    smis2 = ['C(F)(F)C', 'CCCF', 'C(F)C=C']
    smis3 = ['c123c(cc(ccc(N)ccc3)cccc2)cccc1', 'c1cncc1', 'CC(=O)CN']
    n_gram1 = NGram()  # base case
    n_gram1.fit(smis1, train_order=3)
    n_gram2 = NGram()  # higher order but lower num of rings
    n_gram2.fit(smis2, train_order=4)
    n_gram3 = NGram()  # lower order but higher num of rings
    n_gram3.fit(smis3, train_order=2)

    tmp_ngram = n_gram1.merge_table(n_gram2, weight=1, overwrite=False)
    tmp_tab = tmp_ngram._table
    assert (len(tmp_tab), len(tmp_tab[0][0])) == (4,2)

    tmp_ngram = n_gram1.merge_table(n_gram3, weight=1, overwrite=False)
    tmp_tab = tmp_ngram._table
    assert (len(tmp_tab), len(tmp_tab[0][0])) == (3, 4)

    n_gram1.merge_table(n_gram2, n_gram3, weight=[0.5, 1])
    tmp_tab = n_gram1._table
    assert (len(tmp_tab), len(tmp_tab[0][0])) == (4, 4)
    assert tmp_tab[0][0][0].loc["['C']","C"] == 11.0
    assert tmp_tab[0][1][0].loc["['(']", "C"] == 3.0


def test_iqspr_1(data):
    np.random.seed(0)
    ecfp = ECFP(n_jobs=1, input_type='smiles')
    bre = GaussianLogLikelihood(descriptor=ecfp)
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
