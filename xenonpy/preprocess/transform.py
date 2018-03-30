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

    def _check_type(self, x, check_shape=True):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float)
        elif isinstance(x, (DataFrame, Series)):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError(
                'parameter `X` should be a `DataFrame`, `Series`, `ndarray` or list object '
                'but got {}'.format(type(x)))
        if check_shape and x.shape != self._shape:
            raise ValueError('parameter `X` should have shape {} but got {}'.format(
                self._shape, x.shape))
        return x

    def fit(self, x):
        self._shape = x.shape
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
        if len(x.shape) == 1:
            x_, lmd = self._box_cox(col)
            self._lmd.append(lmd)
            return x_

        xs = []
        for col in x.T:
            x_, lmd = self._box_cox(col)
            self._lmd.append(lmd)
            xs.append(x_.reshape(-1, 1))
        df = np.concatenate(xs, axis=1)
        return df

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
        x = self._check_type(x, check_shape=False)
        if len(x.shape) == 1:
            return  inv_boxcox(x, self._lmd[0]) - self._shift + self._min[0]

        xs = []
        for i, col in enumerate(x.T):
            x_ = inv_boxcox(col, self._lmd[i]) - self._shift + self._min[i]
            xs.append(x_.reshape(-1, 1))
        df = np.concatenate(xs, axis=1)
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
        if isinstance(value, (Series, list, np.ndarray)):
            self.__value = DataFrame(data=value)
        elif isinstance(value, DataFrame):
            self.__value = value
        else:
            raise TypeError(
                'value must be list, dict, tuple, Series, ndarray or DataFrame but got {}'.format(type(value)))
        self._index = self.__value.index
        self._columns = self.__value.columns
        self.__now = self.__value.values
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
        return DataFrame(self.__now, index=self._index, columns=self._columns)

    @property
    def value_test(self):
        return self.__now

    def inverse(self, data):
        if len(self.__inverse_chain) == 0:
            return data
        for inv in self.__inverse_chain[::-1]:
            data = inv(data)
        return data

    def reset(self):
        self.__now = self.__value
