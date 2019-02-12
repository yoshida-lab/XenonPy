#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from ..base import BaseLogLikelihood
from ...descriptor.base import BaseDescriptor, BaseFeaturizer


class BayesianRidgeEstimator(BaseLogLikelihood):
    def __init__(self, descriptor):
        self._mdl = BayesianRidge(compute_score=True)
        if not isinstance(descriptor, (BaseFeaturizer, BaseDescriptor)):
            raise TypeError('<descriptor> must be a subclass of <BaseFeaturizer> or <BaseDescriptor>')
        self._descriptor = descriptor

    def fit(self, X, y, **kwargs):
        """
        Parameters
        ----------
        **kwargs
        """
        desc = self._descriptor.transform(X)
        self._mdl.fit(desc, y)

    def log_likelihood(self, smis, target):
        ll = np.repeat(-1000.0, len(smis))
        tar_fps = self._descriptor.transform(smis)
        tmp = tar_fps.isna().any(axis=1)
        idx = [i for i in range(len(smis)) if ~tmp[i]]
        tar_fps.dropna(inplace=True)
        mean, std = self._mdl.predict(tar_fps, return_std=True)
        tmp = norm.logcdf(target, loc=-np.asarray(mean), scale=np.asarray(std))
        np.put(ll, idx, tmp)
        return ll
