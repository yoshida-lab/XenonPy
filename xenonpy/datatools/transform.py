#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import PowerTransformer as PT

from xenonpy.utils import Switch

__all__ = ['PowerTransformer', 'Scaler']


class PowerTransformer(BaseEstimator, TransformerMixin):
    """
    Box-cox transform.
    References
    ----------
    G.E.P. Box and D.R. Cox, “An Analysis of Transformations”,
    Journal of the Royal Statistical Society B, 26, 211-252 (1964).
    """

    def __init__(self,
                 *,
                 method='yeo-johnson',
                 standardize=False,
                 lmd=None,
                 tolerance=(-np.inf, np.inf),
                 on_err=None):
        """

        Parameters
        ----------
        method: 'yeo-johnson' or 'box-cox'
            ‘yeo-johnson’ works with positive and negative values
            ‘box-cox’ only works with strictly positive values
        standardize: boolean
            Normalize to standard normal or not.
            Recommend using a sepearate `standard` function instead of using this option.
        lmd: list or 1-dim ndarray
            You might assign each input xs with a specific lmd yourself.
            Leave None(default) to use a inferred value.
            See `PowerTransformer` for detials.
        tolerance: tuple
            Tolerance of lmd. Set None to accept any.
            Default is **(-np.inf, np.inf)** but recommend **(-2, 2)** for Box-cox transform
        on_err: None or str
            Error handle when try to inference lambda. Can be None or **log**, **nan** or **raise** by string.
            **log** will return the logarithmic transform of xs that have a min shift to 1.
            **nan** return ``ndarray`` with shape xs.shape filled with``np.nan``.
            **raise** raise a FloatingPointError. You can catch it yourself.
            Default(None) will return the input series without scale transform.
        .. _PowerTransformer:
            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer
        """
        self._tolerance = tolerance
        self._pt = PT(method=method, standardize=standardize)
        self._lmd = lmd
        self._shape = None
        self._on_err = on_err

    def _check_type(self, x):
        if isinstance(x, list):
            x = np.array(x, dtype=np.float)
        elif isinstance(x, (DataFrame, Series)):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError(
                'parameter `X` should be a `DataFrame`, `Series`, `ndarray` or list object '
                'but got {}'.format(type(x)))
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        return x

    def fit(self, x):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature transformation

        Returns
        -------
        self : object
            Fitted scaler.
        """

        x = self._pt._check_input(self._check_type(x), in_fit=True)

        # forcing constant column vectors to have no transformation (lambda=1)
        idx = []
        for i, col in enumerate(x.T):
            if np.all(col == col[0]):
                idx.append(i)

        if self._lmd is not None:
            if isinstance(self._lmd, float):
                self._pt.lambdas_ = np.array([self._lmd] * x.shape[1])
            elif x.shape[1] != len(self._lmd):
                raise ValueError('shape[1] of parameter `X` should be {} but got {}'.format(
                    x.shape[1], len(self._lmd)))
            else:
                self._pt.lambdas_ = np.array(self._lmd)
        else:
            self._pt.fit(x)

        if len(idx) > 0:
            self._pt.lambdas_[idx] = 1.

        return self

    def transform(self, x):
        ret = self._pt.transform(self._check_type(x))
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(ret, index=x.index, columns=x.columns)
        return ret

    def inverse_transform(self, x):
        ret = self._pt.inverse_transform(self._check_type(x))
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(ret, index=x.index, columns=x.columns)
        return ret


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

    def power_transformer(self, *args, **kwargs):
        return self._scale(PowerTransformer, *args, **kwargs)

    def box_cox(self, *args, **kwargs):
        return self._scale(PowerTransformer, method='box-cox', *args, **kwargs)

    def yeo_johnson(self, *args, **kwargs):
        return self._scale(PowerTransformer, method='yeo-johnson', *args, **kwargs)

    def min_max(self, *args, **kwargs):
        return self._scale(MinMaxScaler, *args, **kwargs)

    def standard(self, *args, **kwargs):
        return self._scale(StandardScaler, *args, **kwargs)

    def log(self):
        return self._scale(PowerTransformer, method='box-cox', lmd=0.)

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
        if len(self._scalers) == 0:
            return x

        for s in self._scalers:
            x = s.fit_transform(x)
        return self

    def fit_transform(self, x, y=None, **fit_params):
        if len(self._scalers) == 0:
            return x

        x_ = x
        for s in self._scalers:
            x_ = s.fit_transform(x_)

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(x_, index=x.index, columns=x.columns)
        return x_

    def transform(self, x):
        """Scaling features of X according to feature_range.
        Parameters
        ----------
        x : array-like, shape [n_samples, n_features]
            Input data that will be transformed.
        """
        if len(self._scalers) == 0:
            return x

        x_ = x
        for s in self._scalers:
            x_ = s.fit_transform(x_)

        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(x_, index=x.index, columns=x.columns)
        return x_

    def inverse_transform(self, x):
        x_ = x
        for s in self._scalers[::-1]:
            x_ = s.inverse_transform(x_)
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(x_, index=x.index, columns=x.columns)
        return x_

    def reset(self):
        """
        Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        self._scalers = []
