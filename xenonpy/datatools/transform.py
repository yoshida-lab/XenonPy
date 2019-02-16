#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
from pandas import DataFrame, Series
from scipy.special import inv_boxcox, boxcox
from scipy.stats import boxcox as bc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..utils import Switch


class BoxCox(BaseEstimator, TransformerMixin):
    """
    Box-cox transform.

    References
    ----------
    G.E.P. Box and D.R. Cox, “An Analysis of Transformations”,
    Journal of the Royal Statistical Society B, 26, 211-252 (1964).
    """

    def __init__(self, *, lmd=None, shift=1e-9, tolerance=(-2, 2), on_err=None):
        """
        Parameters
        ----------
        lmd: list or 1-dim ndarray
            You might assign each input xs with a specific lmd yourself.
            Leave None(default) to use a inferred value.
            See `boxcox`_ for detials.
        shift: float
            Guarantee Xs are positive.
            BoxCox transform need all data positive.
            Therefore, a shift xs with their min and a specific shift data series(xs)``x = x - x.min + shift``.

        tolerance: tuple
            Tolerance of lmd. Set None to accept any.
            Default is **(-2, 2)**
        on_err: None or str
            Error handle when try to inference lambda. Can be None or **log**, **nan** or **raise** by string.
            **log** will return the logarithmic transform of xs that have a min shift to 1.
            **nan** return ``ndarray`` with shape xs.shape filled with``np.nan``.
            **raise** raise a FloatingPointError. You can catch it yourself.
            Default(None) will return the input series without scale transform.


        .. _boxcox:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
        """
        self._tolerance = tolerance
        self._shift = [shift]
        self._lmd = lmd
        self._shape = None
        self._on_err = on_err

    @property
    def shift_(self):
        return self._shift

    @property
    def lambda_(self):
        return self._lmd

    def _check_type(self, x, check_shape=True):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float)
        elif isinstance(x, (DataFrame, Series)):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError(
                'parameter `X` should be a `DataFrame`, `Series`, `ndarray` or list object '
                'but got {}'.format(type(x)))
        if not self._shape:
            self._shape = x.shape
        if check_shape and len(x.shape) > 1:
            if x.shape[1] != self._shape[1]:
                raise ValueError('parameter `X` should have shape {} but got {}'.format(self._shape, x.shape))
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return x

    def _handle_err(self, e):
        for c in Switch(self._on_err):
            if c(None):
                self._lmd.append(np.inf)
                break
            if c('log'):
                self._lmd.append(0.)
                break
            if c('nan'):
                self._lmd.append(np.nan)
                break
            if c('raise'):
                raise e
            if c():
                raise RuntimeError('parameter on_err must be None "log", "nan" or "raise"')

    def fit(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------

        """
        x = self._check_type(x, check_shape=False)
        if self._lmd is not None:
            if isinstance(self._lmd, float):
                self._lmd = [self._lmd] * x.shape[1]
            if x.shape[1] != len(self._lmd):
                raise ValueError('shape[1] of parameter `X` should be {} but got {}'.format(
                    x.shape[1], len(self._lmd)))
            return self

        self._lmd = []
        self._shift = self._shift * x.shape[1]
        with np.errstate(all='raise'):
            for i, col in enumerate(x.T):
                tmp = col[~np.isnan(col)]
                if not np.all(tmp > 0):
                    tmp = tmp - tmp.min() + self._shift[i]
                try:
                    _, lmd = bc(tmp)
                    if self._tolerance:
                        if not self._tolerance[0] < lmd < self._tolerance[1]:
                            raise FloatingPointError()
                    self._lmd.append(lmd)
                except FloatingPointError as e:
                    self._handle_err(e)

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
        xs = []
        for i, col in enumerate(x.T):
            if np.all(col > 0):
                self._shift[i] = 0.
            else:
                self._shift[i] -= col[~np.isnan(col)].min()

            _lmd = self._lmd[i]
            _shift = self._shift[i]
            for case in Switch(_lmd):
                if case(np.inf):
                    x = col
                    break
                if case(np.nan):
                    x = np.full(col.shape, np.nan)
                    break
                if case():
                    x = boxcox(col + _shift, _lmd)
            xs.append(x.reshape(-1, 1))
        xs = np.concatenate(xs, axis=1)

        if len(self._shape) == 1:
            return xs.ravel()
        return xs.reshape(-1, self._shape[1])

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
        xs = []
        for col, shift, lmd in zip(x.T, self._shift, self._lmd):
            for case in Switch(lmd):
                if case(np.nan, np.inf):
                    _x = col
                    break
                if case():
                    _x = inv_boxcox(col, lmd) - shift
            xs.append(_x.reshape(-1, 1))
        xs = np.concatenate(xs, axis=1)
        if len(self._shape) == 1:
            return xs.ravel()
        return xs


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
        """
        Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        x: array-like, shape [n_samples, n_features]
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        """
        for s in self._scalers:
            x = s.fit_transform(x)
        return self

    def fit_transform(self, x, y=None, **fit_params):
        for s in self._scalers:
            x = s.fit_transform(x)
        return x

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
        """
        Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
