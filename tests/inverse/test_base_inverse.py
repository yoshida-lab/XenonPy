#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import pytest

from xenonpy.inverse.base import BaseLogLikelihood, BaseProposal, BaseResample, BaseSMC


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    class LLH(BaseLogLikelihood):

        def log_likelihood(self, X, **target):
            return pd.DataFrame(np.asanyarray(X))

    class Proposal(BaseProposal):

        def proposal(self, X):
            return X

    class Resample(BaseResample):

        def resample(self, X, freq, size, p):
            return np.random.choice(X, size, p=p)

    class SMC(BaseSMC):

        def __init__(self):
            self._log_likelihood = LLH()
            self._proposal = Proposal()
            self._resample = Resample()

    # prepare test data
    yield dict(llh=LLH, prop=Proposal, smc=SMC, resample=Resample)

    print('test over')


def test_base_loglikelihood_1(data):
    llh = data['llh']()
    X = np.array([1, 2, 3, 3, 4, 4])
    ll = llh(X)
    print(ll)
    assert np.all(ll.values.flatten() == X)


def test_base_proposer_1(data):
    proposer = data['prop']()
    X = np.array([1, 2, 3, 4, 5])
    x = proposer(X)
    assert np.all(x == X)


def test_base_resammple_1(data):
    res = data['resample']()
    X = np.array([1, 2, 3, 4, 5])
    x = res(X, 4, p=[0, 0, 1, 0, 0])
    assert np.all(x == [3, 3, 3, 3])


def test_base_smc_1():
    samples = [1, 2, 3, 4, 5]
    beta = np.linspace(0.05, 1, 5)

    class SMC(BaseSMC):
        pass

    smc = SMC()
    with pytest.raises(NotImplementedError):
        for s in smc(samples, beta=beta):
            assert s


def test_base_smc_2(data):
    class SMC(BaseSMC):
        pass

    smc = SMC()
    llh = data['llh']()
    proposer = data['prop']()
    with pytest.raises(TypeError):
        smc._log_likelihood = 1
    smc._log_likelihood = llh

    with pytest.raises(TypeError):
        smc._proposal = 1
    smc._proposal = proposer


def test_base_smc_3(data):
    smc = data['smc']()

    samples = [1, 2, 100, 4, 5]
    beta = np.linspace(0.05, 1, 5)
    for s, ll, p, f in smc(samples, beta=beta, yield_lpf=True):
        pass
    assert s == 100
    assert np.all(ll == np.array(100))
    assert p == 1.0
    assert f == 5


def test_not_implement():
    base = BaseLogLikelihood()
    with pytest.raises(NotImplementedError):
        base.log_likelihood([1, 2], a=1, b=1)

    base = BaseResample()
    with pytest.raises(NotImplementedError):
        base.resample([1, 2], freq=[5, 5], size=10, p=[.5, .5])

    base = BaseProposal()
    with pytest.raises(NotImplementedError):
        base.proposal([1, 2])


if __name__ == "__main__":
    pytest.main()
