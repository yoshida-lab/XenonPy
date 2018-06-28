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
        self._shift = shift
        self._min = None
        self._lmd = lmd or []
        self._shape = None

    def _check_type(self, x, check_shape=True):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float)
        elif isinstance(x, (DataFrame, Series)):
            x = x.values
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if not isinstance(x, np.ndarray):
            raise TypeError(
                'parameter `X` should be a `DataFrame`, `Series`, `ndarray` or list object '
                'but got {}'.format(type(x)))
        if check_shape:
            if x.shape[1] != self._shape[1]:
                raise ValueError('shape[1] of parameter `X` should be {} but got {}'.format(self._shape[1], x.shape[1]))
        return x

    def _positive(self, x):
        xs = []
        self._min = []
        for col in x.T:
            col_ = x[~np.isnan(x)]
            min_ = col_.min()
            self._min.append(min_)
            tmp = col - min_ + self._shift
            xs.append(tmp)
        return xs

    def fit(self, x):
        x = self._check_type(x, check_shape=False)
        self._shape = x.shape
        if self._lmd:
            if x.shape[1] != len(self._lmd):
                raise ValueError('shape[1] of parameter `X` should be {} but got {}'.format(
                    len(self._lmd), x.shape[1]))
            return self

        for col in x.T:
            col_ = col[~np.isnan(col)]
            if col_.min() != col_.max():
                tmp = col_ - col_.min() + self._shift
                with np.errstate(all='raise'):
                    try:
                        _, lmd = bc(tmp)
                        self._lmd.append(lmd)
                    except FloatingPointError:
                        self._lmd.append(0.)
            else:
                self._lmd.append(0.)

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
        x_ = self._positive(x)
        xs = [boxcox(col, lmd).reshape(-1, 1) for col, lmd in zip(x_, self._lmd)]
        return np.concatenate(xs, axis=1)

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
        x = self._check_type(x, check_shape=True)
        xs = [(inv_boxcox(col, lmd) + min_ - self._shift).reshape(-1, 1) for col, min_, lmd in
              zip(x.T, self._min, self._lmd)]
        return np.concatenate(xs, axis=1)


class Scaler(BaseEstimator, TransformerMixin):
    """
    A value-matrix container for data transform.
    """

    def __init__(self):
        """
        Parameters
        ----------
        value: DataFrame
            Inner data.
        """
        self._scalers = []

    def box_cox(self, *args, **kwargs):
        return self._scale(BoxCox, *args, **kwargs)

    def min_max(self, *args, **kwargs):
        return self._scale(MinMaxScaler, *args, **kwargs)

    def standard(self, *args, **kwargs):
        return self._scale(StandardScaler, *args, **kwargs)

    def log(self):
        return self._scale(BoxCox, lmd=0.)

    def _scale(self, scaler, *args, **kwargs):
        self._scalers.append(scaler(*args, **kwargs))
        return self

    def fit(self, x):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        x: array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """
        for s in self._scalers:
            s.fit(x)
        return self

    def transform(self, x):
        """Scaling features of X according to feature_range.
        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        for s in self._scalers:
            x = s.transform(x)
        return x

    def inverse_transform(self, x):
        for s in self._scalers[::-1]:
            x = s.inverse_transform(x)
        return x

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        self._scalers = []
