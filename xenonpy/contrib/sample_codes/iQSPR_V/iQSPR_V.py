# IQSPR with focus on molecule variety: bring in new initial molecules from reservoir in every step of SMC

import numpy as np
import pandas as pd

from xenonpy.inverse.base import BaseSMC, BaseProposal, BaseLogLikelihood, SMCError

#class SMCError(Exception):
#    """Base exception for SMC classes"""
#    pass

class IQSPR_V(BaseSMC):

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
        self._proposal = modifier
        self._log_likelihood = estimator

    def resample(self, sims, size, p):
        return np.random.choice(sims, size=size, p=p)

    @property
    def modifier(self):
        return self._proposal

    @modifier.setter
    def modifier(self, value):
        self._proposal = value

    @property
    def estimator(self):
        return self._log_likelihood

    @estimator.setter
    def estimator(self, value):
        self._log_likelihood = value
        
    def __call__(self, reservoir, beta, size=100, *, samples=None, ratio=0.5, yield_lpf=False):
        """
        Run SMC
        Parameters
        ----------
        reservoir: list of object
            Samples to be drawn as new initial molecules in each step of SMC
        beta: list/1D-numpy of float or pd.Dataframe
            Annealing parameters for each step.
            If pd.Dataframe, column names should follow keys of mdl in BaseLogLikeihood or BaseLogLikelihoodSet
        size: int
            Sample size for each draw.
        samples: list of object
            Initial samples.
        ratio: float
            ratio of molecules to be replaced from reservoir in each step of SMC
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

        # initial samples will be randomly picked from resevoir if samples are provided
        if samples is None:
            samples = np.random.choice(reservoir,size=size).tolist()
        # refill samples if len(samples) not equals given size
        elif len(samples) < size:
            samples = np.concatenate([np.array(samples), np.random.choice(reservoir,size=size-len(samples))])
            
        res_size = int(size*ratio)
        smc_size = size - res_size

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
                re_samples = self.resample(unique, smc_size, p)
                samples = self.proposal(np.concatenate([re_samples, np.random.choice(reservoir,size=res_size)]))

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
                
