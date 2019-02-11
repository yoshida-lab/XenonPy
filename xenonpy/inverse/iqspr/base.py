#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator


class BaseLogLikelihood(BaseEstimator, ABC):

    def fit(self, X, y, **kwargs):
        return self

    @abstractmethod
    def predict(self, X):
        """"""
        pass


class BaseModifier(BaseEstimator, ABC):
    def fit(self, X, y, **kwargs):
        return self

    @abstractmethod
    def transform(self, X):
        """"""
        pass
