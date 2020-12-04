#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from copy import deepcopy
from sklearn.linear_model import BayesianRidge

from xenonpy.descriptor import ECFP, RDKitFP
from xenonpy.inverse.iqspr import GaussianLogLikelihood, NGram, IQSPR, IQSPR4DF, GetProbError, MolConvertError
from xenonpy.inverse.base import BaseLogLikelihoodSet


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
    rdkitfp = RDKitFP(n_jobs=1, input_type='smiles')
    bre = GaussianLogLikelihood(descriptor=ecfp)
    bre2 = GaussianLogLikelihood(descriptor=rdkitfp)
    bre.fit(X, y[['bandgap', 'glass_transition_temperature']])
    bre2.fit(X, y[['density', 'refractive_index']])
    bre.update_targets(bandgap=(1, 2), glass_transition_temperature=(200, 300))
    bre2.update_targets(refractive_index=(2, 3), density=(0.9, 1.2))

    class MyLogLikelihood(BaseLogLikelihoodSet):
        def __init__(self):
            super().__init__()

            self.loglike = bre
            self.loglike = bre2

    like_mdl = MyLogLikelihood()
    ngram = NGram()
    ngram.fit(X[0:20], train_order=5)
    iqspr = IQSPR(estimator=bre, modifier=ngram)
    # prepare test data
    yield dict(ecfp=ecfp, rdkitfp=rdkitfp, bre=bre, bre2=bre2, like_mdl=like_mdl, ngram=ngram, iqspr=iqspr, pg=(X, y))

    print('test over')


def test_gaussian_ll_1(data):
    bre = deepcopy(data['bre'])
    bre2 = data['bre2']
    X, y = data['pg']

    assert 'bandgap' in bre._mdl
    assert 'glass_transition_temperature' in bre._mdl
    assert 'refractive_index' in bre2._mdl
    assert 'density' in bre2._mdl

    ll = bre.log_likelihood(X.sample(10),
                            bandgap=(7, 8),
                            glass_transition_temperature=(300, 400))
    assert ll.shape == (10,2)
    assert isinstance(bre['bandgap'], BayesianRidge)
    assert isinstance(bre['glass_transition_temperature'], BayesianRidge)

    with pytest.raises(KeyError):
        bre['other']

    with pytest.raises(TypeError):
        bre['other'] = 1
    bre['other'] = BayesianRidge()

    bre.remove_estimator()
    assert bre._mdl == {}


def test_gaussian_ll_2(data):
    bre = deepcopy(data['bre'])
    X, y = data['pg']

    x = X.sample(10)
    tmp = bre.predict(x)
    assert isinstance(tmp, pd.DataFrame)
    assert tmp.shape == (10, 4)

    x[66666] = 'CCd'
    tmp = bre.predict(x)
    assert isinstance(tmp, pd.DataFrame)
    assert tmp.shape == (11, 4)
    print(tmp)
    tmp = tmp.loc[66666]

    assert tmp.isna().all()


def test_gaussian_ll_3(data):
    like_mdl = data['like_mdl']
    X, y = data['pg']
    ll = like_mdl.log_likelihood(X.sample(10))

    assert ll.shape == (10,4)


def test_gaussian_ll_4(data):
    # check if training of NaN data and pd.Series input are ok
    ecfp = data['ecfp']
    bre = GaussianLogLikelihood(descriptor=ecfp)
    train_data = pd.DataFrame({'x': ['C','CC','CCC','CCCC','CCCCC'], 'a': [np.nan, np.nan, 3, 4, 5], 'b': [1, 2, 3, np.nan, np.nan]})
    bre.fit(train_data['x'], train_data['a'])

    bre.remove_estimator()
    bre.fit(train_data['x'], train_data[['a','b']])


def test_ngram_1(data):
    ngram = NGram()
    assert ngram.ngram_table is None
    assert ngram.max_len == 1000
    assert ngram.del_range == (1, 10)
    assert ngram.reorder_prob == 0
    assert ngram.sample_order == (1, 10)
    assert ngram._train_order is None

    ngram.set_params(max_len=500, reorder_prob=0.2)

    assert ngram.max_len == 500
    assert ngram.del_range == (1, 10)
    assert ngram.reorder_prob == 0.2


def test_ngram_2(data):
    ngram = NGram()

    with pytest.warns(RuntimeWarning, match='<sample_order>'):
        ngram.fit(data['pg'][0][:20], train_order=5)

    assert ngram._train_order == (1, 5)
    assert ngram.sample_order == (1, 5)
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
    assert n_gram1._train_order == (1, 3)

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
    assert tmp_ngram._train_order == (1, 4)
    tmp_tab = tmp_ngram._table
    assert (len(tmp_tab), len(tmp_tab[0][0])) == (4,2)

    tmp_ngram = n_gram1.merge_table(n_gram3, weight=1, overwrite=False)
    assert tmp_ngram._train_order == (1, 3)
    tmp_tab = tmp_ngram._table
    assert (len(tmp_tab), len(tmp_tab[0][0])) == (3, 4)

    n_gram1.merge_table(n_gram2, n_gram3, weight=[0.5, 1])
    tmp_tab = n_gram1._table
    assert n_gram1._train_order == (1, 4)
    assert (len(tmp_tab), len(tmp_tab[0][0])) == (4, 4)
    assert tmp_tab[0][0][0].loc["['C']","C"] == 11.0
    assert tmp_tab[0][1][0].loc["['(']", "C"] == 3.0


def test_ngram_6(data):
    smis0 = ['CCCc1ccccc1', 'CC(CCc1ccccc1)CC', 'Cc1ccccc1CC', 'C(CC(C))CC', 'CCCC']
    n_gram0 = NGram()  # base case
    n_gram0.fit(smis0, train_order=5)

    n_gram1, n_gram2 = n_gram0.split_table(cut_order=2)
    assert n_gram1._train_order == (1, 2)
    assert n_gram1.min_len == 1
    assert n_gram2._train_order == (3, 5)
    assert n_gram2.min_len == 3

    n_gram3 = n_gram2.merge_table(n_gram1, weight=1, overwrite=False)
    assert n_gram3._train_order == (1, 5)
    assert n_gram3.min_len == 3
    assert np.all(n_gram3._table[3][0][1] == n_gram0._table[3][0][1])
    assert np.all(n_gram3._table[2][1][0] == n_gram0._table[2][1][0])
    n_gram1.merge_table(n_gram2, weight=1)
    assert n_gram1._train_order == (1, 5)
    assert n_gram1.min_len == 1
    assert np.all(n_gram1._table[3][0][1] == n_gram0._table[3][0][1])
    assert np.all(n_gram1._table[2][1][0] == n_gram0._table[2][1][0])


def test_iqspr_1(data):
    np.random.seed(0)
    ecfp = data['ecfp']
    bre = GaussianLogLikelihood(descriptor=ecfp)
    ngram = NGram()
    iqspr = IQSPR(estimator=bre, modifier=ngram)
    X, y = data['pg']
    bre.fit(X, y)
    bre.update_targets(reset=True, bandgap=(0.1, 0.2), density=(0.9, 1.2))
    ngram.fit(data['pg'][0][0:20], train_order=10)
    beta = np.linspace(0.05, 1, 10)
    for s, ll, p, f in iqspr(data['pg'][0][:5], beta, yield_lpf=True):
        assert np.abs(np.sum(p) - 1.0) < 1e-5
        assert np.sum(f) == 5


def test_iqspr_2(data):
    np.random.seed(0)
    like_mdl = data['like_mdl']
    ngram = data['ngram']
    iqspr = IQSPR(estimator=like_mdl, modifier=ngram)

    beta1 = np.linspace(0.05, 1, 10)
    beta2 = np.linspace(0.01, 1, 10)
    beta = pd.DataFrame({'bandgap': beta1, 'glass_transition_temperature': beta2,
                        'density': beta1, 'refractive_index': beta2})
    for s, ll, p, f in iqspr(data['pg'][0][:5], beta, yield_lpf=True):
        assert np.abs(np.sum(p) - 1.0) < 1e-5
        assert np.sum(f) == 5


def test_iqspr_resample1(data):
    # not sure if this test can be fully reliable by only fixing the random seed
    like_mdl = data['like_mdl']
    ngram = data['ngram']
    beta = np.linspace(0.1, 1, 2)

    np.random.seed(0)
    iqspr = IQSPR(estimator=like_mdl, modifier=ngram, r_ESS=0)
    soln1 = [['C([*])C([*])(C(=O)OCCSCCC#N)', 'C([*])C([*])(SCCC)',
             'O([*])C(=O)OC(C=C1)=CC=C1C(C=C2)=CC=C2CC(C=C3)=CC=C3C(C=C4)=CC=C4([*])'],
            ['C([*])C([*])(C(=O)OCC(F)(F)C(F)(F)OC(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)F)',
             'C([*])C([*])(CC)(C(=O)OCC(F)(F)F)',
    'O([*])C(=O)OC(C=C1)=CC=C1C(C=C1)=CC=C1CC(C=C1)=CC=C1C(C=C1)=CC=C1C(C=C1)=CC=C1C(C=C1)=CC=C1C(C=C1)=CC=C1C(=S)']
            ]
    c0 = 0
    for s, ll, p, f in iqspr(data['pg'][0][:3], beta, yield_lpf=True):
        assert np.abs(np.sum(p) - 1.0) < 1e-5
        assert np.sum(f) == 3
        assert np.all(np.sort(s) == np.array(soln1[c0]))
        c0 += 1

    np.random.seed(0)
    iqspr = IQSPR(estimator=like_mdl, modifier=ngram, r_ESS=1)
    soln2 = [['C([*])C([*])(C(=O)OCCSCCC#N)', 'C([*])C([*])(SCCC)',
             'O([*])C(=O)OC(C=C1)=CC=C1C(C=C2)=CC=C2CC(C=C3)=CC=C3C(C=C4)=CC=C4([*])'],
            ['O([*])C(=O)OC(C=C1)=CC=C1C(C=C1)=CC=C1CC(C=C1)=CC=C1C(C=C1)=CC=C1([*])',
             'O([*])C(=O)OC(C=C1)=CC=C1C(C=C1)=CC=C1CC(C=C1)=CC=C1C(C=C1)=CC=C1C(=S)']
            ]
    c0 = 0
    for s, ll, p, f in iqspr(data['pg'][0][:3], beta, yield_lpf=True):
        assert np.abs(np.sum(p) - 1.0) < 1e-5
        assert np.sum(f) == 3
        assert np.all(np.sort(s) == np.array(soln2[c0]))
        c0 += 1


def test_iqspr4df_unique1(data):
    # not sure if this test can be fully reliable by only fixing the random seed
    like_mdl = data['like_mdl']
    ngram = data['ngram']
    beta = np.linspace(0.1, 1, 1)
    samples = pd.DataFrame([data['pg'][0][:2].values.repeat(2), [0, 1, 2, 3]]).T

    np.random.seed(0)
    iqspr = IQSPR4DF(estimator=like_mdl, modifier=ngram, r_ESS=0, sample_col=0)
    soln = pd.DataFrame([['C([*])C([*])(SCCC)', 'C([*])C([*])(C(=O)OCCSCCC#N)'], [0, 2]]).T
    for s, ll, p, f in iqspr(samples, beta, yield_lpf=True):
        assert np.abs(np.sum(p) - 1.0) < 1e-5
        assert np.sum(f) == 4
        assert (s == soln).all().all()


if __name__ == "__main__":
    pytest.main()
