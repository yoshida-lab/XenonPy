#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
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

        def log_likelihood(self, X, **targets):
            return np.sum(np.asanyarray([X * (j - i) for i, j in targets.values()]), axis=0)

    class Proposal(BaseProposal):

        def proposal(self, X):
            return X

    class Resample(BaseResample):

        def resample(self, X, size, p):
            return np.random.choice(X, size, p=p)

    class SMC(BaseSMC):

        def __init__(self):
            self._log_likelihood = LLH()
            self._proposal = Proposal()
            self._resample = Resample()

    # prepare test data
    yield dict(llh=LLH, prop=Proposal, smc=SMC, resample=Resample)

    print('test over')


def test_base_loglikelihood1(data):
    llh = data['llh']()
    X = np.array([1, 2, 3, 3, 4, 4])
    ll = llh(X, tar1=(5, 10), tar2=(1, 5))
    assert np.all(ll == np.array([9, 18, 27, 27, 36, 36]))


def test_base_proposer1(data):
    proposer = data['prop']()
    X = np.array([1, 2, 3, 4, 5])
    x = proposer(X)
    assert np.all(x == X)


def test_base_resammple1(data):
    res = data['resample']()
    X = np.array([1, 2, 3, 4, 5])
    x = res(X, 4, p=[0, 0, 1, 0, 0])
    assert np.all(x == [3, 3, 3, 3])


def test_base_smc1():
    samples = [1, 2, 3, 4, 5]
    beta = np.linspace(0.05, 1, 5)

    class SMC(BaseSMC):
        pass

    smc = SMC()
    try:
        for s in smc(samples, beta=beta):
            assert s
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'
    assert not smc.targets

    try:
        for s in smc(samples, beta=beta, tar1=(5, 10), tar2=(1, 5)):
            assert s

    except NotImplementedError:
        assert True
    else:
        assert False, 'should got NotImplementedError'
    assert smc.targets

    class SMC(BaseSMC):

        def log_likelihood(self, X):
            return np.sum(np.asanyarray([X * (j - i) for i, j in self._targets.values()]), axis=0)

        def proposal(self, X):
            return X

        def resample(self, X, size, p):
            return np.random.choice(X, size, p=p)

    smc = SMC()
    assert not smc.targets
    try:
        for s in smc(samples, beta=beta):
            assert s
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'


def test_base_smc2(data):
    class SMC(BaseSMC):
        pass

    smc = SMC()
    llh = data['llh']()
    proposer = data['prop']()
    try:
        smc._log_likelihood = 1
    except TypeError:
        assert True
    else:
        assert False, 'should got TypeError'

    try:
        smc._log_likelihood = llh
    except TypeError:
        assert False, 'should got TypeError'
    else:
        assert True

    try:
        smc._proposal = 1
    except TypeError:
        assert True
    else:
        assert False, 'should got TypeError'

    try:
        smc._proposal = proposer
    except TypeError:
        assert False, 'should got TypeError'
    else:
        assert True


def test_base_smc3(data):
    smc = data['smc']()

    samples = [1, 2, 100, 4, 5]
    beta = np.linspace(0.05, 1, 5)
    for s, ll, p, f in smc(samples, beta=beta, tar1=(5, 10), tar2=(1, 5), yield_lpf=True):
        pass
    assert s == 100
    assert ll == 900
    assert p == 1.0
    assert f == 5


if __name__ == "__main__":
    pytest.main()
