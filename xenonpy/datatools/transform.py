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

    def __init__(self, *, lmd=None, shift=1e-9, tolerance=(-2, 2), on_err=None):
        """
        Parameters
        ----------
        lmd: list or 1-dim ndarray
            You might assign each input x a lmd by yourself.
            Leave None(default) to use a inference value.
            See `boxcox`_ for detials.
        shift: float
            Guarantee Xs positive. ``x = x - x.min + shift``
        tolerance: tuple
            Tolerance of lmd. Set None to accept any.
        on_err: None or 'log', 'nan' and 'raise' in string
            Error handle. Default is None means return input series.
            Can be set to ``log``, ``nan`` or ``raise``.


        .. _boxcox:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
        """
        self._tolerance = tolerance
        self._shift = shift
        self._lmd = lmd
        self._min = []
        self._shape = None
        self._on_err = on_err

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

    def _positive(self, x):
        xs = []
        for i, col in enumerate(x.T):
            _lmd = self._lmd[i]
            if _lmd == 0. or _lmd is np.inf:
                xs.append(col)
                self._min.append(0.)
            elif _lmd is np.nan:
                xs.append(np.array([np.nan] * len(col)))
                self._min.append(0.)
            else:
                col_ = col[~np.isnan(col)]
                min_ = col_.min()
                self._min.append(min_)
                xs.append(col - min_ + self._shift)
        return xs

    def _handle_err(self, e):
        if self._on_err is None:
            self._lmd.append(np.inf)
        elif self._on_err is 'log':
            self._lmd.append(0.)
        elif self._on_err is 'nan':
            self._lmd.append(np.nan)
        elif self._on_err is 'raise':
            raise e
        else:
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
        with np.errstate(all='raise'):
            for col in x.T:
                col_ = col[~np.isnan(col)]
                tmp = col_ - col_.min() + self._shift
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
        x_ = self._positive(x)
        xs = []
        for col, lmd in zip(x_, self._lmd):
            if lmd is np.nan or lmd is np.inf:
                xs.append(col.reshape(-1, 1))
            else:
                xs.append(boxcox(col, lmd).reshape(-1, 1))
        xs = np.concatenate(xs, axis=1)
        return xs.reshape(*self._shape)

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
        for col, min_, lmd in zip(x.T, self._min, self._lmd):
            if lmd is np.nan or lmd is np.inf:
                xs.append(col.reshape(-1, 1))
            elif lmd == 0.:
                xs.append(np.exp(col).reshape(-1, 1))
            else:
                xs.append((inv_boxcox(col, lmd) + min_ - self._shift).reshape(-1, 1))
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
