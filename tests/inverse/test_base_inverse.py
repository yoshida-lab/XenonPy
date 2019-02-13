#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pytest

from xenonpy.inverse.base import BaseLogLikelihood, BaseProposal, BaseSMC


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    class LLH(BaseLogLikelihood):
        def log_likelihood(self, X, target):
            return X * target

    class Proposal(BaseProposal):
        def proposal(self, X, size, *, p=None):
            return np.random.choice(X, size=size, p=p)

    class SMC(BaseSMC):
        def __init__(self):
            self._log_likelihood = LLH()
            self._proposal = Proposal()

    # prepare test data
    yield dict(llh=LLH, prop=Proposal, smc=SMC)

    print('test over')


def test_base_loglikelihood1(data):
    llh = data['llh']()
    X = np.array([1, 2, 3, 3, 4, 4])
    ll = llh(X, 10)
    assert np.all(ll == np.array([1, 2, 3, 3, 4, 4]) * 10)


def test_base_proposer1(data):
    proposer = data['prop']()
    X = np.array([1, 2, 3, 4, 5])
    x = proposer(X, 4, p=[0, 0, 1, 0, 0])
    assert np.all(x == [3, 3, 3, 3])


def test_base_smc1():
    class SMC(BaseSMC):
        pass

    smc = SMC()
    samples = [1, 2, 3, 4, 5]
    beta = np.linspace(0.05, 1, 100)
    try:
        for s in smc(samples, beta=beta):
            pass

    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'

    smc._target = 400
    try:
        for s in smc(samples, beta=beta):
            pass

    except NotImplementedError:
        assert True
    else:
        assert False, 'should got NotImplementedError'


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
    for s, ll, p, f in smc(samples, beta=beta, target=1, yield_lpf=True):
        pass
    assert s[0] == 100
    assert ll[0] == 100
    assert p[0] == 1.0
    assert f[0] == 5


if __name__ == "__main__":
    pytest.main()
