# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
from pandas import DataFrame, Series
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class BoxCox(BaseEstimator, TransformerMixin):
    """
    Box-cox transform.

    References
    ----------
    G.E.P. Box and D.R. Cox, “An Analysis of Transformations”,
    Journal of the Royal Statistical Society B, 26, 211-252 (1964).
    """

    def __init__(self, shift=1e-9):
        """
        Parameters
        ----------
        shift: float
            Guarantee that all variables > 0
        """
        self._shift = shift
        self._min = []
        self._lmd = []
        self._shape = None

    def fit(self, x):
        self._shape = x.shape
        if len(self._shape) == 1:
            self._shape = (self._shape[0], 1)
        return self

    def transform(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------
        DataFrame
            Box-Cox transformed data.
        """
        x = self._check_type(x)
        df = DataFrame()
        for col in x.columns:
            x_, lmd = self._box_cox(x[col])
            self._lmd.append(lmd)
            df[col] = x_
        return df

    def _check_type(self, x):
        if isinstance(x, (np.ndarray, Series, list)):
            x = DataFrame(data=x)
        if not isinstance(x, DataFrame):
            raise TypeError(
                'parameter `X` should be a `DataFrame`, `Series`, `ndarray` or list object'
                'but got {}'.format(type(x)))
        if x.shape != self._shape:
            raise ValueError('parameter `X` should have shape {} but got {}'.format(
                self._shape, x.shape))
        return x

    def _box_cox(self, series):
        if series.min() != series.max():
            self._min.append(series.min())
            with np.errstate(all='raise'):
                tmp = series - series.min() + self._shift
                try:
                    return boxcox(tmp)
                except FloatingPointError:
                    return boxcox(tmp, 0.), 0.
        self._min.append(series.min())
        tmp = series - series.min() + self._shift
        return boxcox(tmp, 0.), 0.

    def inverse_transform(self, x):
        """
        Scale back the data to the original representation.

        Parameters
        ----------
        x: DataFrame, Series, ndarray, list
            The data used to scale along the features axis.

        Returns
        -------
        DataFrame
            Inverse transformed data.
        """
        x = self._check_type(x)
        df = DataFrame()
        for i, col in enumerate(x.columns):
            x_ = inv_boxcox(x[col], self._lmd[i]) - self._shift + self._min[i]
            df[col] = x_
        return df


class Scaler(object):
    """
    A value-matrix container for data transform.
    """

    def __init__(self, value):
        """
        Parameters
        ----------
        value: DataFrame
            Inner data.
        """
        if isinstance(value, DataFrame):
            self.__value = value
        elif isinstance(value, (list, dict, tuple, Series, np.ndarray)):
            self.__value = DataFrame(data=value)
        else:
            raise TypeError(
                'value must be list, dict, tuple, Series, ndarray or DataFrame but got {}'.format(type(value)))
        self.__now = self.__value
        self.__inverse_chain = []

    def box_cox(self, *args, **kwargs):
        return self._scale(BoxCox, *args, **kwargs)

    def min_max(self, *args, **kwargs):
        return self._scale(MinMaxScaler, *args, **kwargs)

    def standard_scale(self, *args, **kwargs):
        return self._scale(StandardScaler, *args, **kwargs)

    def _scale(self, scaler, *args, **kwargs):
        scaler = scaler(*args, **kwargs)
        self.__now = scaler.fit_transform(self.__now)
        self.__inverse_chain.append(scaler.inverse_transform)
        return self

    @property
    def value(self):
        return DataFrame(self.__now, index=self.__value.index, columns=self.__value.columns)

    def inverse(self, data):
        if len(self.__inverse_chain) == 0:
            return data
        for inv in self.__inverse_chain[::-1]:
            data = inv(data)
        return data

    def reset(self):
        self.__now = self.__value
