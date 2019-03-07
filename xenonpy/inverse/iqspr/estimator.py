#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from copy import deepcopy
from types import MethodType

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.linear_model import BayesianRidge

from ..base import BaseLogLikelihood
from ...descriptor.base import BaseDescriptor, BaseFeaturizer


class BayesianRidgeEstimator(BaseLogLikelihood):
    def __init__(self, descriptor, **estimators):
        """
        Bayesian Ridge Estimator.

        Parameters
        ----------
        descriptor: BaseFeaturizer or BaseDescriptor
            Descriptor generator.
        estimators: BaseEstimator
            Bayesian estimators in scikit-learn style.
            When pass ``return_std=True`` to Estimator's ``predict`` method,
            ``y_std`` should be returned at second. e.g: ``BayesianRidge`` in scikit-learn.
        """
        if estimators:
            self._mdl = deepcopy(estimators)
        else:
            self._mdl = {}
        if not isinstance(descriptor, (BaseFeaturizer, BaseDescriptor)):
            raise TypeError('<descriptor> must be a subclass of <BaseFeaturizer> or <BaseDescriptor>')
        self._descriptor = descriptor

    @property
    def estimators(self):
        return self._mdl

    def __getitem__(self, item):
        return self._mdl[item]

    def __setitem__(self, key, value):
        if not (hasattr(value, 'predict') and isinstance(value.predict, MethodType)):
            raise TypeError('estimator must be a regressor in scikit-learn style')
        self._mdl[key] = deepcopy(value)

    # todo: implement scale function
    def fit(self, smiles, y=None, *, X_scaler=None, y_scaler=None, **kwargs):
        """
        Parameters
        ----------
        smiles: list[str]
            SMILES for training.
        y: pandas.DataFrame
            Target properties for training.
        X_scaler: Scaler (optional, not implement)
            Scaler for transform X.
        y_scaler: Scaler (optional, not implement)
            Scaler for transform y.
        kwargs: dict
            Parameters pass to BayesianRidge initialization.
        """

        if not isinstance(y, pd.DataFrame):
            raise TypeError('please package all properties into a pd.DataFrame')

        # remove NaN fromm X
        desc = self._descriptor.transform(smiles)
        desc = pd.DataFrame(data=desc).reset_index(drop=True)
        y = y.reset_index(drop=True)
        desc.dropna(inplace=True)
        y = y.loc[desc.index]

        for c in y:
            y_ = y[c]  # get target property.
            # remove NaN from y_
            y_.dropna(inplace=True)
            desc_ = desc.loc[y_.index]
            desc_ = desc_.values

            mdl = BayesianRidge(compute_score=True, **kwargs)
            mdl.fit(desc_, y_)
            self._mdl[c] = mdl

    def log_likelihood(self, smis, **targets):
        def _avoid_overflow(ll_):
            # log(exp(log(UP) - log(C)) - exp(log(LOW) - log(C))) + log(C)
            # where C = max(log(UP), max(LOW))
            ll_c = np.max(ll_)
            ll_ = np.log(np.exp(ll_[1] - ll_c) - np.exp(ll_[0] - ll_c)) + ll_c
            return ll_

        ll = np.repeat(-1000.0, len(smis))
        tar_fps = self._descriptor.transform(smis)
        tar_fps = pd.DataFrame(data=tar_fps).reset_index(drop=True)
        tmp = tar_fps.isna().any(axis=1)
        idx = [i for i in range(len(smis)) if ~tmp[i]]
        tar_fps.dropna(inplace=True)

        # calculate likelihood
        ll_mat = []
        for k, (low, up) in targets.items():  # k: target; v: (low, up)

            # predict mean, std for all smiles
            mean, std = self._mdl[k].predict(tar_fps, return_std=True)

            # calculate low likelihood
            low_ll = norm.logcdf(low, loc=np.asarray(mean), scale=np.asarray(std))

            # calculate up likelihood
            up_ll = norm.logcdf(up, loc=np.asarray(mean), scale=np.asarray(std))

            # zip low and up likelihood to a 1-dim array then save it.
            # like: [(tar_low_smi1, tar_up_smi1),  (tar_low_smi2, tar_up_smi2), ..., (tar_low_smiN, tar_up_smiN)]
            lls = zip(low_ll, up_ll)
            ll_mat.append(list(lls))

        # sum all ll along each smiles
        # ll_sum = [[sum_low_smi1, sum_up_smi1], [sum_low_smi2, sum_up_smi2],...,[sum_low_smiN, sum_up_smiN],]
        ll_sum = np.sum(np.array(ll_mat), axis=0)
        tmp = np.array([*map(_avoid_overflow, ll_sum)])

        np.put(ll, idx, tmp)
        return ll
