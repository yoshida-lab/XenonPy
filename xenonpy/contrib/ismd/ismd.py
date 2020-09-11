#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd

from xenonpy.inverse.base import BaseSMC, BaseProposal, BaseLogLikelihood


class ISMD(BaseSMC):

    def __init__(self, *, estimator, modifier):
        """
        SMC iqspr runner.

        Parameters
        ----------
        estimator : BaseLogLikelihood or BaseLogLikelihoodSet
            Log likelihood estimator for given SMILES.
        modifier : BaseProposal
            Modify given SMILES to new ones.
        """
        self._proposal = modifier
        self._log_likelihood = estimator

    def resample(self, samples, size, p):
        resample = samples.sample(n=size, replace=True, weights=p)["reactant_index"]
        return pd.DataFrame(resample).reset_index(drop=True)

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

    def unique(self, samples):
        """

        Parameters
        ----------
        X: list of object
            Input samples.

        Returns
        -------
        unique: list of object
            The sorted unique values.
        unique_counts: np.ndarray of int
            The number of times each of the unique values comes up in the original array
        """
        unique_reactant, frequency = np.unique(samples["reactant_index"], return_counts=True)
        unique_samples = pd.DataFrame({"reactant_index": unique_reactant})
        return unique_samples, frequency
