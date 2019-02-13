#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# import necessary libraries

from ..base import BaseSMC, BaseProposer, BaseLogLikelihood


class IQSPR(BaseSMC):

    def __init__(self, *, target, estimator, modifier):
        """
        SMC iQDPR runner.

        Parameters
        ----------
        estimator : BaseLogLikelihood
        modifier : BaseProposer
        """
        self._proposer = modifier
        self._log_likelihood = estimator
        self._target = target

    @property
    def modifier(self):
        return self._proposer

    @modifier.setter
    def modifier(self, value):
        self._proposer = value

    @property
    def estimator(self):
        return self._log_likelihood

    @estimator.setter
    def estimator(self, value):
        self._log_likelihood = value
