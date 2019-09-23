#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


from collections import defaultdict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from xenonpy.utils import TimedMetaClass


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
        log_likelihood: pd.Dataframe of float (col - properties, row - samples)
            Estimated log-likelihood of each sample's property values.
            Cannot be pd.Series!
        """
        # raise NotImplementedError('<log_likelihood> have no implementation')
        raise NotImplementedError('<log_likelihood> method must be implemented')


class BaseLogLikelihoodSet(BaseEstimator, metaclass=TimedMetaClass):
    """
    Abstract class to organize log-likelihoods.
    
    Examples
    --------
    .. code::

        class MyLogLikelihood(BaseLogLikelihoodSet):
            def __init__(self):
                super().__init__()
    
                self.loglike1 = SomeFeature1()
                self.loglike1 = SomeFeature2()
                self.loglike2 = SomeFeature3()
                self.loglike2 = SomeFeature4()

    """

    def __init__(self, *, loglikelihoods='all'):
        """
            
        Parameters
        ----------
        loglikelihoods: list[str] or 'all'
            log-likelihoods that will be used.
            Default is 'all'.
        """
        self.loglikelihoods = loglikelihoods
        self.__loglikelihoods__ = set()
        self.__loglikelihood_sets__ = defaultdict(list)

    @property
    def elapsed(self):
        return self._timer.elapsed

    def __setattr__(self, key, value):

        if key == '__loglikelihood_sets__':
            if not isinstance(value, defaultdict):
                raise RuntimeError('Can not set "self.__loglikelihood_sets__" by yourself')
            super().__setattr__(key, value)
        if isinstance(value, BaseLogLikelihood):
            #            if value.__class__.__name__ in self.__loglikelihoods__:
            #                raise RuntimeError('Duplicated log-likelihood <%s>' % value.__class__.__name__)
            self.__loglikelihood_sets__[key].append(value)
            self.__loglikelihoods__.add(value.__class__.__name__)
        else:
            super().__setattr__(key, value)

    @property
    def all_loglikelihoods(self):
        return list(self.__loglikelihoods__)

    def _check_input(self, X, y=None, **kwargs):
        def _reformat(x):
            if x is None:
                return x

            keys = list(self.__loglikelihood_sets__.keys())
            if len(keys) == 1:
                if isinstance(x, list):
                    return pd.DataFrame(pd.Series(x), columns=keys)

                if isinstance(x, np.ndarray):
                    if len(x.shape) == 1:
                        return pd.DataFrame(x, columns=keys)

                if isinstance(x, pd.Series):
                    return pd.DataFrame(x.values, columns=keys, index=x.index)

            if isinstance(x, pd.Series):
                x = pd.DataFrame(x)

            if isinstance(x, pd.DataFrame):
                tmp = set(x.columns) | set(kwargs.keys())
                if set(keys).isdisjoint(tmp):
                    raise KeyError(
                        'name of columns do not match any log-likelihood set')
                return x

            raise TypeError(
                'you can not ues a array-like input'
                'because there are multiply log-likelihood sets or the dim of input is not 1')

        return _reformat(X), _reformat(y)

    def __call__(self, X, **kwargs):
        return self.log_likelihood(X, **kwargs)

    def log_likelihood(self, X, **kwargs):
        """
        Log likelihood
        
        Parameters
        ----------
        X: list[object]
            Input samples for likelihood calculation.
        kwargs: list[string]
            specified BaseLogLikelihood.
        
        Returns
        -------
        log_likelihood: pd.Dataframe of float (col - properties, row - samples)
            Estimated log-likelihood of each sample's property values.
        """
        # raise NotImplementedError('<log_likelihood> have no implementation')
        # raise NotImplementedError('<log_likelihood> method must be implemented')

        if not isinstance(X, Iterable):
            raise TypeError('parameter "entries" must be a iterable object')

        if len(X) is 0:
            return None

        if 'loglikelihoods' in kwargs:
            loglikelihoods = kwargs['loglikelihoods']
            if loglikelihoods != 'all' and not isinstance(loglikelihoods, list):
                raise TypeError('parameter "loglikelihoods" must be a list')
        else:
            loglikelihoods = self.loglikelihoods

        # if 'return_type' in kwargs:
        #    del kwargs['return_type']

        results = []

        X, _ = self._check_input(X, **kwargs)
        for k, lls in self.__loglikelihood_sets__.items():
            if k in kwargs:
                k = kwargs[k]  # what is this for? even if k not in kwargs... nothing happen right?
            if k in X:
                for f in lls:
                    if loglikelihoods != 'all' and f.__class__.__name__ not in loglikelihoods:
                        continue
                    ret = f.log_likelihood(X[k])
                    results.append(ret)

        return pd.concat(results, axis=1)


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

    # _targets = None

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
        # return self._log_likelihood(X, **self._targets)
        return self._log_likelihood(X)

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

    def on_errors(self, ite, samples, error):
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
        if key is '_log_likelihood' and not isinstance(value, (BaseLogLikelihood, BaseLogLikelihoodSet)):
            raise TypeError(
                '<self._log_likelihood> must be a subClass of <BaseLogLikelihood> or <BaseLogLikelihoodSet>')
        if key is '_proposal' and not isinstance(value, BaseProposal):
            raise TypeError('<self._proposal> must be a subClass of <BaseProposal>')
        if key is '_resample' and not isinstance(value, BaseResample):
            raise TypeError('<self._resample> must be a subClass of <BaseResample>')
        object.__setattr__(self, key, value)

    def __call__(self, samples, beta, *, size=None, yield_lpf=False):
        """
        Run SMC

        Parameters
        ----------
        samples: list of object
            Initial samples.
        beta: list/1D-numpy of float or pd.Dataframe
            Annealing parameters for each step.
            If pd.Dataframe, column names should follow keys of mdl in BaseLogLikeihood or BaseLogLikelihoodSet
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

        #         if targets:
        #             self.update_targets(**targets)
        #         else:
        #             if not self._targets:
        #                 raise ValueError('must set targets area')

        try:
            unique, frequency = self.unique(samples)
            ll = self.log_likelihood(unique)

            if isinstance(beta, pd.DataFrame):
                beta = beta[ll.columns].values
            else:
                # assume only one row for beta (1-D list or numpy vector)
                beta = np.transpose(np.repeat([beta], ll.shape[1], axis=0))

            w = np.dot(ll.values, beta[0]) + np.log(frequency)
            w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
            p = np.exp(w - w_sum)
            if yield_lpf:
                yield unique, ll, p, frequency
            else:
                yield unique

            # beta = np.delete(beta,0,0) #remove first "row"

        except SMCError as e:
            self.on_errors(0, samples, e)
        except Exception as e:
            raise e

        # translate between input representation and execute environment representation.
        # make sure beta is not changed (np.delete will return the deleted version, without changing original vector)
        for i, step in enumerate(np.delete(beta, 0, 0)):
            try:
                re_samples = self.resample(unique, size, p)
                samples = self.proposal(re_samples)

                unique, frequency = self.unique(samples)

                # annealed likelihood in log - adjust with copy counts
                ll = self.log_likelihood(unique)
                w = np.dot(ll.values, step) + np.log(frequency)
                w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
                p = np.exp(w - w_sum)
                if yield_lpf:
                    yield unique, ll, p, frequency
                else:
                    yield unique

            except SMCError as e:
                self.on_errors(i + 1, samples, e)
            except Exception as e:
                raise e
