#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# import necessary libraries
import numpy as np
import scipy.stats as sps
from rdkit import Chem

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

    def translator(self, x, inverse=False):
        if inverse:
            return [self.esmi2smi(x_) for x_ in x]
        return [self.smi2esmi(x_) for x_ in x]

    def log_likelihood(self, X):
        # target molecule with Tg > tar_min_Tg

        ll = np.repeat(-1000.0, len(X))  # note: the constant will determine the vector type!
        mols = []
        idx = []
        for i in range(len(X)):
            try:
                mols.append(Chem.MolFromSmiles(self.esmi2smi(X[i])))
                idx.append(i)
            except BaseException:
                pass
        # convert extended SMILES to fingerprints
        tar_fps = self.descriptor_gen.proposal(mols)
        tmp = tar_fps.isna().any(axis=1)
        idx = [idx[i] for i in range(len(idx)) if ~tmp[i]]
        tar_fps.dropna(inplace=True)
        # predict Tg values and calc. log-likelihood
        tar_mean, tar_std = self._estimator.log_likelihood(tar_fps, target=)
        tmp = sps.norm.logcdf(-self.target, loc=-np.asarray(tar_mean), scale=np.asarray(tar_std))
        np.put(ll, idx, tmp)
        return ll

    def proposal(self, X, size, p=None):
        if self.n_gram_table is None:
            raise ValueError(
                'Must have a pre-trained n-gram table,',
                'you can set one your already had or train a new one by using <update_ngram> method'
            )
        pass
