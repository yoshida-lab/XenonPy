#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
from sklearn.base import BaseEstimator

from ..utils import TimedMetaClass


class BaseSMC(BaseEstimator, metaclass=TimedMetaClass):

    def log_likelihood(self, x):
        """
        Likelihood function.

        Parameters
        ----------
        x: list of object
            Samples for likelihood calculation.

        Returns
        -------
        loglikelihood: list of float
            Log scaled likelihood values.
        """
        raise NotImplementedError('<likelihood> method must be implemented')

    def proposal(self, x, size, p=None):
        """

        Parameters
        ----------
        x: list of object
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
        raise NotImplementedError('<next> method must be implemented')

    def translator(self, x, inverse=False):
        """
        Translate between input representation and execute environment representation.

        Parameters
        ----------
        x: list of object
            Objects to translate.
        inverse: bool
            If ``True``, do a inverse translation.

        Returns
        -------
        list of object
            The translated objects.
        """

        if inverse:
            return x
        return x

    def unique(self, x):
        """
        Count unique values.

        Parameters
        ----------
        x: list of object
            Input array. This should be flattened if it is not already 1-D.

        Returns
        -------
        unique: list of object
            The sorted unique values.
        unique_counts: list of int
            The number of times each of the unique values comes up in the original array.
        """
        return np.unique(x, return_counts=True)

    def __call__(self, samples, beta, size=None):
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

        Yields
        -------
        samples: list of object
            New samples in each SMC iteration.
        frequency: list of int
            The frequency corresponding to samples.
        """

        # sample size will be set to the length of init_samples if None
        if size is None:
            size = len(samples)
        samples, frequency = self.unique(samples)

        # translate between input representation and execute environment representation.
        samples = self.translator(samples)
        for i, step in enumerate(beta):
            # annealed likelihood in log - adjust with copy counts
            ll = self.log_likelihood(samples)
            w = ll * step + np.log(frequency)
            wSum = np.log(sum(np.exp(w - max(w)))) + max(w)  # avoid underflow
            probs = np.exp(w - wSum)
            samples = self.proposal(samples, size, p=probs)
            samples, frequency = self.unique(samples)

            yield self.translator(samples, inverse=True), frequency, w
