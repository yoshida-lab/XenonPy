#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# import necessary libraries

import numpy as np
import pandas as pd
from xenonpy.inverse.base import BaseSMC, BaseProposal, BaseLogLikelihood


class IQSPR4DF(BaseSMC):

    def __init__(self, *, estimator, modifier, r_ESS=1, sample_col=None):
        """
        SMC iqspr runner (assume data type of samples = pd.DataFrame).

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
        sample_col : list or str
            Name(s) of columns that will be used to extract unique samples in the unique function.
            Default is None, which means all columns are used.
        """
        self._proposal = modifier
        self._log_likelihood = estimator
        self._r_ESS = r_ESS
        if isinstance(sample_col, str):
            self.sample_col = [sample_col]
        elif hasattr(sample_col, '__len__'):
            self.sample_col = sample_col
        else:
            self.sample_col = [sample_col]

    def resample(self, sims, freq, size, p):
        if np.sum(np.power(p, 2)) <= (self._r_ESS*np.sum(freq)):
            return sims.sample(n=size, replace=True, weights=p).reset_index(drop=True)
        else:
            return sims.loc[sims.index.repeat(freq), :].reset_index(drop=True)

    def unique(self, x):
        """

        Parameters
        ----------
        X: pd.DataFrame
            Input samples.

        Returns
        -------
        unique: pd.DataFrame
            The sorted unique samples.
        unique_counts: np.ndarray of int
            The number of times each of the unique values comes up in the original array
        """

        if self.sample_col is None:
            sample_col = x.columns.values
        else:
            sample_col = self.sample_col
        uni = x.drop_duplicates(subset=sample_col).reset_index(drop = True)
        freq = []
        for index,row in uni.iterrows():
            tar = row[sample_col]
            x_ = x
            for c,t in zip(sample_col,tar):
                x_ = x_.loc[x_[c] == t]
            freq.append(len(x_))
        return uni, freq

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
