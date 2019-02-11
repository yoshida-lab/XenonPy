#  Copyright 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from ..utils import TimedMetaClass


class BaseLogLikelihood(BaseEstimator, ABC):

    def fit(self, X, y=None, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        return self.log_likelihood(*args, **kwargs)

    @abstractmethod
    def log_likelihood(self, X, *, target):
        """"""
        pass


class BaseProposer(BaseEstimator, ABC):
    def fit(self, X, y, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        return self.proposal(*args, **kwargs)

    @abstractmethod
    def proposal(self, X, size, p=None):
        """"""
        pass


class BaseSMC(BaseEstimator, metaclass=TimedMetaClass):
    _log_likelihood = None
    _proposer = None

    def log_likelihood(self, X, *, target):
        """
        Likelihood function.

        Parameters
        ----------
        target
        X: list of object
            Samples for likelihood calculation.

        Returns
        -------
        log_likelihood: list of float
            Log scaled likelihood values.
        frequency: list of float
            Number of times each of the unique sample comes up in the original samples.
        """
        if self._log_likelihood is None:
            raise NotImplementedError('user need to implement <likelihood> method'
                                      'or set <self._log_likelihood> to a instance of <BaseLogLikelihood>')
        return self._log_likelihood(X, target=target)

    def proposal(self, X, size, p=None):
        """

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
        if self._proposer is None:
            raise NotImplementedError('user need to implement <proposal> method'
                                      'or set <self._proposer> to a instance of <BaseProposer>')
        return self._proposer(X, size, p)

    def on_errors(self, ite, samples, target, error):
        raise error

    def __setattr__(self, key, value):
        if key is '_log_likelihood' and not isinstance(value, BaseLogLikelihood):
            raise TypeError('must be a subClass of <BaseLogLikelihood>')
        if key is '_proposer' and not isinstance(value, BaseProposer):
            raise TypeError('must be a subClass of <BaseProposer>')
        object.__setattr__(self, key, value)

    def __call__(self, samples, beta, target, size=None, yield_llh=False, yield_weight=False):
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
        yield_llh : bool
            Yield estimated log likelihood of each samples. Default is ``False``.
        yield_weight : bool
            Yield estimated weight of each samples. Default is ``False``.

        Yields
        -------
        samples: list of object
            New samples in each SMC iteration.
        llh: list of float
            Estimated log likelihood of each samples.
            Only yield when ``yield_llh=Ture``.
        weight: list of float
            Estimated weight of each samples.
            Only yield when ``yield_weight=Ture``.
        """

        # sample size will be set to the length of init_samples if None
        if size is None:
            size = len(samples)

        # translate between input representation and execute environment representation.
        for i, step in enumerate(beta):
            try:
                # annealed likelihood in log - adjust with copy counts
                ll, frequency = self.log_likelihood(samples, target=target)
                w = ll * step + np.log(frequency)
                wSum = np.log(sum(np.exp(w - max(w)))) + max(w)  # avoid underflow
                probs = np.exp(w - wSum)
                samples = self.proposal(samples, size, p=probs)
            except BaseException as e:
                self.on_errors(i, samples, target, e)

            tmp = (samples,)
            if yield_llh:
                tmp += (ll,)
            if yield_weight:
                tmp += (w,)
            if len(tmp) > 1:
                yield tmp
            else:
                yield samples
