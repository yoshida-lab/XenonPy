#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# import necessary libraries

import numpy as np

from ..base import BaseSMC, BaseProposal, BaseLogLikelihood


class IQSPR(BaseSMC):

    def __init__(self, *, target, estimator, modifier):
        """
        SMC iqspr runner.

        Parameters
        ----------
        estimator : BaseLogLikelihood
        modifier : BaseProposal
        """
        self._proposal = modifier
        self._log_likelihood = estimator
        self._target = target

    def resample(self, X, size, p):
        return np.random.choice(sims, size=size, p=p)

    @property
    def modifier(self):
        return self._proposal

    @modifier.setter
    def modifier(self, value):
        self._proposal = value

    @property
    def estimator(self):
        return self._log_likelihood

    @estimator.setter
    def estimator(self, value):
        self._log_likelihood = value
