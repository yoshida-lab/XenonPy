#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from sklearn.linear_model import BayesianRidge

from .base import BaseLogLikelihood


class BayesianRidgeEstimator(BaseLogLikelihood):
    def __init__(self):
        self.mdl = BayesianRidge(compute_score=True)

    def fit(self, X, y):
        """"""
        self.mdl.fit(X, y)

    def predict(self, X):
        return self.mdl.predict(X, return_std=True)
