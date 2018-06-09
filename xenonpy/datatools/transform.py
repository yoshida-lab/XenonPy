# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
from pandas import DataFrame, Series
from scipy.special import inv_boxcox, boxcox
from scipy.stats import boxcox as bc
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

    def __init__(self, *, lmd=None, shift=1e-9):
        """
        Parameters
        ----------
        shift: float
            Guarantee that all variables > 0
        """
        self.shift = shift
        self.lmd = lmd
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
            x_, lmd = self._box_cox(x)
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
        _series = series[~np.isnan(series)]

        if self.lmd is not None:
            _min = _series.min()
            self._min.append(_min)
            tmp = series - _min + self.shift
            return boxcox(tmp, self.lmd), self.lmd

        if _series.min() != _series.max():
            _min = _series.min()
            self._min.append(_min)
            tmp = _series - _min + self.shift
            with np.errstate(all='raise'):
                try:
                    _, lmd = bc(tmp)
                    return boxcox(series - _min + self.shift, lmd), lmd
                except FloatingPointError:
                    return boxcox(series - _min + self.shift, 0.), 0.

        _min = _series.min()
        self._min.append(_min)
        tmp = series - _min + self.shift
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
            return inv_boxcox(x, self._lmd[0]) - self.shift + self._min[0]

        xs = []
        for i, col in enumerate(x.T):
            x_ = inv_boxcox(col, self._lmd[i]) - self.shift + self._min[i]
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
            self._value = DataFrame(data=value)
        elif isinstance(value, DataFrame):
            self._value = value
        else:
            raise TypeError(
                'value must be list, dict, tuple, Series, ndarray or DataFrame but got {}'.format(type(value)))
        self._index = self._value.index
        self._columns = self._value.columns
        self._now = self._value.values
        self._inverse_chain = []

    def box_cox(self, *args, **kwargs):
        return self._scale(BoxCox, *args, **kwargs)

    def min_max(self, *args, **kwargs):
        return self._scale(MinMaxScaler, *args, **kwargs)

    def standard(self, *args, **kwargs):
        return self._scale(StandardScaler, *args, **kwargs)

    def log(self):
        return self._scale(BoxCox, lmd=0.)

    def _scale(self, scaler, *args, **kwargs):
        scaler = scaler(*args, **kwargs)
        self._now = scaler.fit_transform(self._now)
        self._inverse_chain.append(scaler.inverse_transform)
        return self

    @property
    def data_frame(self):
        """
        Return scaled values as Dataframe object.

        Returns
        -------
        DataFrame
            Scaled value. If your need value as ndarray object, please use ``np_value``
        """
        return DataFrame(self._now, index=self._index, columns=self._columns)

    @property
    def values(self):
        """
        Return scaled values as ndarray object

        Returns
        -------
        ndarray
            Scaled value. If your need value as Dataframe object, please use ``value``
        """
        return self._now

    def inverse(self, data):
        if len(self._inverse_chain) == 0:
            return data
        for inv in self._inverse_chain[::-1]:
            data = inv(data)
        return data

    def reset(self):
        self._now = self._value
