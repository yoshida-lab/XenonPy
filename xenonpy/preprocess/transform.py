# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
from pandas import DataFrame, Series
from scipy.special import inv_boxcox
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin


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
            raise ValueError('parameter `X` should have shape {}'.format(
                self._shape))
        return x

    def _box_cox(self, series):
        if series.min() != series.max():
            self._min.append(series.min())
            with np.errstate(all='raise'):
                tmp = series - series.min() + self._shift
                try:
                    return boxcox(tmp)
                except FloatingPointError:
                    return series, None

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
