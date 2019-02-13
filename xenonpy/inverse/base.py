#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from ..utils import TimedMetaClass


class BaseLogLikelihood(BaseEstimator, ABC):

    def fit(self, X, y=None, **kwargs):
        return self

    def __call__(self, X, target):
        return self.log_likelihood(X, target)

    @abstractmethod
    def log_likelihood(self, X, target):
        """
        Log likelihood

        Parameters
        ----------
        X: list of object
            Input samples for likelihood calculation.
        target: float
            Target area.

        Returns
        -------
        log_likelihood: np.ndarray of float
            Estimated likelihood of each samples.
        """
        pass


class BaseProposal(BaseEstimator, ABC):
    def fit(self, X, y, **kwargs):
        return self

    def __call__(self, X, size, *, p=None):
        return self.proposal(X, size, p=p)

    @abstractmethod
    def proposal(self, X, size, *, p=None):
        """
        Proposal new samples based on the input samples.

        Parameters
        ----------
        X: list of object
            Samples for generate next samples
        size: int
            Sampling size.
        p: list of float
            A 1-D array-like object.
            The probabilities associated with each entry in x. Should be a 1-D array-like object.

        Returns
        -------
        samples: list of object
            Generated samples from input samples.
        """
        pass


class BaseSMC(BaseEstimator, metaclass=TimedMetaClass):
    _log_likelihood = None
    _proposal = None
    _target = None

    def log_likelihood(self, X, target):
        """
        Likelihood function.

        Parameters
        ----------
        X: list of object
            Samples for likelihood calculation.
        target : float
            Search Target
        Returns
        -------
        log_likelihood: np.ndarray of float
            Log scaled likelihood values.
        """
        if self._log_likelihood is None:
            raise NotImplementedError('user need to implement <likelihood> method'
                                      'or set <self._log_likelihood> to a instance of <BaseLogLikelihood>')
        return self._log_likelihood(X, target)

    def proposal(self, X, size, *, p=None):
        """
        Proposal new samples based on the input samples.

        Parameters
        ----------
        X: list of object
            Samples for generate next samples
        size: int
            Sampling size.
        p: list of float
            A 1-D array-like object.
            The probabilities associated with each entry in x. Should be a 1-D array-like object.

        Returns
        -------
        samples: list of object
            Generated samples from input samples.
        """
        if self._proposal is None:
            raise NotImplementedError('user need to implement <proposal> method'
                                      'or set <self._proposal> to a instance of <BaseProposal>')
        return self._proposal(X, size, p=p)

    def on_errors(self, ite, samples, target, error):
        raise error

    def unique(self, X):
        """

        Parameters
        ----------
        X: list of object
            Input samples.

        Returns
        -------
        unique: list of object
            The sorted unique values.
        unique_counts: np.ndarray of int
            The number of times each of the unique values comes up in the original array
        """
        return np.unique(X, return_counts=True)

    def __setattr__(self, key, value):
        if key is '_log_likelihood' and not isinstance(value, BaseLogLikelihood):
            raise TypeError('must be a subClass of <BaseLogLikelihood>')
        if key is '_proposal' and not isinstance(value, BaseProposal):
            raise TypeError('must be a subClass of <BaseProposal>')
        object.__setattr__(self, key, value)

    def __call__(self, samples, beta, *, target=None, size=None, yield_lpf=False):
        """
        Run SMC

        Parameters
        ----------
        samples: list of object
            Initial samples.
        beta: list of float
            Annealing parameters for each step.
            Should be a 1-D array-like object.
        target : float
            Search Target
        size: int
            Sample size for each draw.
        yield_lpf : bool
            Yield estimated log likelihood, probability and frequency of each samples. Default is ``False``.

        Yields
        -------
        samples: list of object
            New samples in each SMC iteration.
        llh: np.ndarray float
            Estimated values of log-likelihood of each samples.
            Only yield when ``yield_lpf=Ture``.
        p: np.ndarray of float
            Estimated probabilities of each samples.
            Only yield when ``yield_lpf=Ture``.
        freq: np.ndarray of float
            The number of unique samples in original samples.
            Only yield when ``yield_lpf=Ture``.
        """

        # sample size will be set to the length of init_samples if None
        if size is None:
            size = len(samples)

        if target is not None:
            self._target = target
        else:
            if self._target is None:
                raise ValueError('must set a <target>')

        # translate between input representation and execute environment representation.
        for i, step in enumerate(beta):
            unique, frequency = self.unique(samples)
            try:
                # annealed likelihood in log - adjust with copy counts
                ll = self.log_likelihood(unique, self._target)
                w = ll * step + np.log(frequency)
                w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
                p = np.exp(w - w_sum)
                if yield_lpf:
                    yield unique, ll, p, frequency
                else:
                    yield unique
                samples = self.proposal(unique, size, p=p)
            except BaseException as e:
                self.on_errors(i, samples, target, e)

        try:
            unique, frequency = self.unique(samples)
            if yield_lpf:
                ll = self.log_likelihood(unique, self._target)
                w = ll + np.log(frequency)
                w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
                p = np.exp(w - w_sum)
                yield unique, ll, p, frequency
            else:
                yield unique
        except BaseException as e:
            self.on_errors(i + 1, samples, target, e)
