import numpy as np
import pandas as pd

from xenonpy.inverse.base import BaseSMC, BaseProposal, BaseLogLikelihood


class SMCError(Exception):
    """Base exception for SMC classes"""
    pass


class IQSPR4DF(BaseSMC):

    def __init__(self, *, estimator, modifier):
        """
        SMC iqspr runner.
        Parameters
        ----------
        estimator : BaseLogLikelihood or BaseLogLikelihoodSet
            Log likelihood estimator for given SMILES.
        modifier : BaseProposal
            Modify given SMILES to new ones.
        """
        self.proposal = modifier
        self.log_likelihood = estimator

    @property
    def modifier(self):
        return self.proposal

    @modifier.setter
    def modifier(self, value):
        self.proposal = value

    @property
    def estimator(self):
        return self.log_likelihood

    @estimator.setter
    def estimator(self, value):
        self.log_likelihood = value

    def resample(self, sims, size, p):
        return sims.sample(n=size, replace=True, weights=p).reset_index(drop=True)

    def unique(self, X):
        """
        Parameters
        ----------
        X: pandas.dataframe
            Input samples.
        Returns
        -------
        unique: list of object
            The sorted unique values.
        unique_counts: np.ndarray of int
            The number of times each of the unique values comes up in the original array
        """
        X_unique = X.drop_duplicates(subset=self._sample_col, keep='first').reset_index(drop=True)
        count_dic = X[self._sample_col].value_counts().reindex(index=X_unique[self._sample_col])
        return X_unique, list(count_dic)

    def __call__(self, samples, beta, *, size=None, sample_col=None, yield_lpf=False):
        """
        Run SMC
        Parameters
        ----------
        samples: pandas.dataframe
            Initial samples.
        beta: list/1D-numpy of float or pd.Dataframe
            Annealing parameters for each step.
            If pd.Dataframe, column names should follow keys of mdl in BaseLogLikeihood or BaseLogLikelihoodSet
        size: int
            Sample size for each draw.
        sample_col: list
            list of column names treated as samples (used for unique function)
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
            size = samples.shape[0]

        if sample_col is None:
            self._sample_col = samples.columns.tolist()
        else:
            self._sample_col = [sample_col]
        samples = samples.reset_index(drop=True)

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
