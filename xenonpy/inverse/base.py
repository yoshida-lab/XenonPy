#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator

from ..utils import TimedMetaClass


class LogLikelihoodError(Exception):
    """Base exception for LogLikelihood classes"""
    pass


class ResampleError(Exception):
    """Base exception for Resample classes"""
    pass


class ProposalError(Exception):
    """Base exception for Proposal classes"""
    old_smi = None


class SMCError(Exception):
    """Base exception for SMC classes"""
    pass


class BaseLogLikelihood(BaseEstimator, metaclass=TimedMetaClass):

    def fit(self, X, y, **kwargs):
        return self

    def __call__(self, X, **targets):
        return self.log_likelihood(X, **targets)

    def log_likelihood(self, X, **targets):
        """
        Log likelihood

        Parameters
        ----------
        X: list[object]
            Input samples for likelihood calculation.
        targets: tuple[float, float]
            Target area.
            Should be a tuple which have down and up boundary.
            e.g: ``target1=(10, 20)`` equal to ``target1 should in range [10, 20]``.

        Returns
        -------
        log_likelihood: np.ndarray of float
            Estimated likelihood of each samples.
        """
        # raise NotImplementedError('<log_likelihood> have no implementation')
        raise NotImplementedError('<log_likelihood> method must be implemented')


class BaseResample(BaseEstimator, metaclass=TimedMetaClass):

    def fit(self, X, y=None, **kwargs):
        return self

    def __call__(self, X, size, p):
        return self.resample(X, size, p)

    def resample(self, X, size, p):
        """
        Re-sample from given samples.

        Parameters
        ----------
        X: list of object
            Input samples for likelihood calculation.
        size: int
            Resample size.
        p: np.ndarray of float
            The probabilities associated with each entry in X.
            If not given the sample assumes a uniform distribution over all entries.

        Returns
        -------
        new_sample: list of object
            Re-sampling result.
        """
        raise NotImplementedError('<resample> method must be implemented')


class BaseProposal(BaseEstimator, metaclass=TimedMetaClass):
    def fit(self, X, y, **kwargs):
        return self

    def __call__(self, X):
        return self.proposal(X)

    def on_errors(self, error):
        raise error

    def proposal(self, X):
        """
        Proposal new samples based on the input samples.

        Parameters
        ----------
        X: list of object
            Samples for generate next samples

        Returns
        -------
        samples: list of object
            Generated samples from input samples.
        """
        raise NotImplementedError('<proposal> method must be implemented')


class BaseSMC(BaseEstimator, metaclass=TimedMetaClass):
    _log_likelihood = None
    _proposal = None
    _resample = None
    _targets = None

    def log_likelihood(self, X):
        """
        Log likelihood

        Parameters
        ----------
        X: list of object
            Input samples for likelihood calculation.

        Returns
        -------
        log_likelihood: np.ndarray of float
            Estimated likelihood of each samples.
        """
        if self._log_likelihood is None:
            raise NotImplementedError('user need to implement <log_likelihood> method or'
                                      'set <self._log_likelihood> to a instance of <BaseLogLikelihood>')
        return self._log_likelihood(X, **self._targets)

    def resample(self, X, size, p):
        """
        Re-sample from given samples.

        Parameters
        ----------
        X: list[object]
            Input samples for likelihood calculation.
        size: int
            Resample size.
        p: numpy.ndarray[float]
            The probabilities associated with each entry in X.
            If not given the sample assumes a uniform distribution over all entries.

        Returns
        -------
        re-sample: list of object
            Re-sampling result.
        """
        if self._resample is None:
            raise NotImplementedError('user need to implement <resample> method or'
                                      'set <self._resample> to a instance of <BaseResample>')
        return self._resample(X, size, p)

    def proposal(self, X):
        """
        Proposal new samples based on the input samples.

        Parameters
        ----------
        X: list of object
            Samples for generate next samples

        Returns
        -------
        samples: list of object
            Generated samples from input samples.
        """
        if self._proposal is None:
            raise NotImplementedError('user need to implement <proposal> method or'
                                      'set <self._proposal> to a instance of <BaseProposal>')
        return self._proposal(X)

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

    @property
    def targets(self):
        return deepcopy(self._targets)

    def update_targets(self, *, reset=False, **targets):
        """
        Update/set the target area.

        Parameters
        ----------
        reset: bool
            If ``true``, reset target area.
        targets: tuple[float, float]
            Target area.
            Should be a tuple which have down and up boundary.
            e.g: ``target1=(10, 20)`` equal to ``target1 should in range [10, 20]``.

        """
        if not self._targets or reset:
            self._targets = {}
        for k, v in targets.items():
            if not isinstance(v, tuple) or len(v) != 2 or v[1] <= v[0]:
                raise ValueError('must be a tuple with (low, up) boundary')
            self._targets[k] = v

    def __setattr__(self, key, value):
        if key is '_log_likelihood' and not isinstance(value, BaseLogLikelihood):
            raise TypeError('<self._log_likelihood> must be a subClass of <BaseLogLikelihood>')
        if key is '_proposal' and not isinstance(value, BaseProposal):
            raise TypeError('<self._proposal> must be a subClass of <BaseProposal>')
        if key is '_resample' and not isinstance(value, BaseResample):
            raise TypeError('<self._resample> must be a subClass of <BaseResample>')
        object.__setattr__(self, key, value)

    def __call__(self, samples, beta, *, size=None, yield_lpf=False, **targets):
        """
        Run SMC

        Parameters
        ----------
        samples: list of object
            Initial samples.
        beta: list of float
            Annealing parameters for each step.
            Should be a 1-D array-like object.
        size: int
            Sample size for each draw.
        yield_lpf : bool
            Yield estimated log likelihood, probability and frequency of each samples. Default is ``False``.
        targets: tuple[float, float]
            Target area.
            Should be a tuple which have down and up boundary.
            e.g: ``target1=(10, 20)`` equal to ``target1 should in range [10, 20]``.

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

        if targets:
            self.update_targets(**targets)
        else:
            if not self._targets:
                raise ValueError('must set targets area')

        # translate between input representation and execute environment representation.
        for i, step in enumerate(beta):
            unique, frequency = self.unique(samples)
            try:
                # annealed likelihood in log - adjust with copy counts
                ll = self.log_likelihood(unique)
                w = ll * step + np.log(frequency)
                w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
                p = np.exp(w - w_sum)
                if yield_lpf:
                    yield unique, ll, p, frequency
                else:
                    yield unique

                re_samples = self.resample(unique, size, p)
                samples = self.proposal(re_samples)
            except SMCError as e:
                self.on_errors(i, samples, targets, e)
            except Exception as e:
                raise e

        try:
            unique, frequency = self.unique(samples)
            if yield_lpf:
                ll = self.log_likelihood(unique)
                w = ll + np.log(frequency)
                w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
                p = np.exp(w - w_sum)
                yield unique, ll, p, frequency
            else:
                yield unique
        except SMCError as e:
            self.on_errors(i + 1, samples, targets, e)
        except Exception as e:
            raise e
