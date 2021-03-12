#  Copyright (c) 2021. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# import necessary libraries

import numpy as np

from xenonpy.inverse.base import BaseSMC, BaseProposal, BaseLogLikelihood


class IQSPR(BaseSMC):

    def __init__(self, *, estimator, modifier, r_ESS=1):
        """
        SMC iqspr runner (assume data type of samples = list or np.array).

        Parameters
        ----------
        estimator : BaseLogLikelihood or BaseLogLikelihoodSet
            Log likelihood estimator for given input samples.
        modifier : BaseProposal
            Modify given input samples to new ones.
        r_ESS : float
            r_ESS*sample_size = Upper threshold of ESS (effective sample size) using in SMC resampling.
            Resample will happen only if calculated ESS is smaller or equal to the upper threshold.
            As 1 <= ESS <= sample_size, picking any r_ESS < 1/sample_size will lead to never resample;
            picking any r_ESS >= 1 will lead to always resample.
            Default is 1, i.e., resample at each step of SMC.
        """
        self._proposal = modifier
        self._log_likelihood = estimator
        self._r_ESS = r_ESS

    def resample(self, sims, freq, size, p):
        if np.sum(np.power(p, 2)) <= (self._r_ESS*np.sum(freq)):
            return np.random.choice(sims, size=size, p=p)
        else:
            return [item for item, count in zip(sims, freq) for i in range(count)]

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
