#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# import necessary libraries

import numpy as np

from xenonpy.inverse.base import BaseSMC, BaseProposal, BaseLogLikelihood


class ISMD(BaseSMC):

    def __init__(self, *, estimator, modifier, reactor):
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
        self._reactor = reactor

    def __call__(self, init_reac,target_range):
        
        print("Generation: 0")
        _,product = self._reactor.react(init_reac)
        tmp_ll = self.estimator(product, **target_range)
        tmp = tmp_ll.sum(axis = 1, skipna = True)
        w = np.dot(tmp.values, 1)
        w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
        p = np.exp(w - w_sum)
        selected = np.random.choice(init_reac, size=len(init_reac), replace=True, p=p)
        new_reac = self._proposal.proposal(selected)
        yield init_reac, product, tmp_ll
        
        for i in range(1,5):
            init_reac = new_reac
            print("Generation: "+str(i))
            _,product = self._reactor.react(init_reac)
            tmp_ll = self.estimator(product, **target_range)
            tmp = tmp_ll.sum(axis = 1, skipna = True)
            w = np.dot(tmp.values, 1)
            w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
            p = np.exp(w - w_sum)
            selected = np.random.choice(init_reac, size=len(init_reac), replace=True, p=p)
            new_reac = self._proposal.proposal(selected)

            yield init_reac, product, tmp_ll
        

    def resample(self, sims, size, p):
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
