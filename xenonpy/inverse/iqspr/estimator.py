#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from inverse.base import BaseLogLikelihood


class BayesianRidgeEstimator(BaseLogLikelihood):
    def __init__(self):
        self.mdl = BayesianRidge(compute_score=True)

    def fit(self, X, y, **kwargs):
        """
        Parameters
        ----------
        **kwargs
        """
        self.mdl.fit(X, y)

    def log_likelihood(self, smiles, *, target):
        unique, frequency = np.unique(smiles, return_counts=True)
        mean, std = self.mdl.predict(smiles, return_std=True)
        ll = norm.logcdf(target, loc=-np.asarray(mean), scale=np.asarray(std))
        return ll, frequency
