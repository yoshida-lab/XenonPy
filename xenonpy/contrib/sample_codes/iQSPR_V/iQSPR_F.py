# IQSPR with focus on molecule variety: bring in new initial molecules from reservoir in every step of SMC

import numpy as np
import pandas as pd

from rdkit import Chem
from xenonpy.inverse.iqspr import NGram
from xenonpy.inverse.base import BaseSMC, BaseProposal, BaseLogLikelihood, SMCError

# class SMCError(Exception):
#     """Base exception for SMC classes"""
#     pass

class IQSPR_F(BaseSMC):

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
        
    def fragmenting(self, smis):
        frag_list = []
        
        for smi in smis:
            decomp = Chem.Recap.RecapDecompose(Chem.MolFromSmiles(smi))
            first_gen = [node.mol for node in decomp.children.values()]
            frag_list += [Chem.MolToSmiles(x) for x in first_gen]
                
        return list(set(frag_list))
    
    def combine_fragments(self, smis_base, smis_frag):
        """
        combine two SMILES strings with '*' as connection points
        Parameters
        ----------
        smis_base: str
            SMILES for combining.
            If no '*', assume connection point at the end.
            If more than one '*', the first will be picked if it's not the 1st character.
        smis_frag: str
            SMILES for combining.
            If no '*', assume connection point at the front.
            If more than one '*', the first will be picked.
        """

        # prepare NGram object for use of ext. SMILES
        ngram = NGram()

        # check position of '*'
        mols_base = Chem.MolFromSmiles(smis_base)
        if mols_base is None:
            raise RuntimeError('Invalid base SMILES!')
        idx_base = [i for i in range(mols_base.GetNumAtoms()) if mols_base.GetAtomWithIdx(i).GetSymbol() == '*']

        # rearrange base SMILES to avoid 1st char = '*'
        if len(idx_base) == 1 and idx_base[0] == 0:
            smis_base_head = Chem.MolToSmiles(mols_base,rootedAtAtom=1)
        elif len(idx_base) == 0:
            smis_base_head = smis_base + '*'
        else:
            smis_base_head = smis_base

        # converge base to ext. SMILES and pick insertion location
        esmi_base = ngram.smi2esmi(smis_base_head)
        esmi_base = esmi_base[:-1]
        idx_base = esmi_base.index[esmi_base['esmi'] == '*'].tolist()
        if idx_base[0] == 0:
            idx_base = idx_base[1]
        else:
            idx_base = idx_base[0]

        # rearrange fragment to have 1st char = '*' and convert to ext. SMILES
        mols_frag = Chem.MolFromSmiles(smis_frag)
        if mols_frag is None:
            raise RuntimeError('Invalid frag SMILES!')
        idx_frag = [i for i in range(mols_frag.GetNumAtoms()) if mols_frag.GetAtomWithIdx(i).GetSymbol() == '*']
        if len(idx_frag) == 0:
            esmi_frag = ngram.smi2esmi(smis_frag)
            # remove last '!'
            esmi_frag = esmi_frag[:-1]
        else:
            esmi_frag = ngram.smi2esmi(Chem.MolToSmiles(mols_frag,rootedAtAtom=idx_frag[0]))
            # remove leading '*' and last '!'
            esmi_frag = esmi_frag[1:-1]

        # check open rings of base SMILES
        nRing_base = esmi_base['n_ring'].loc[idx_base]

        # re-number rings in fragment SMILES
        esmi_frag['n_ring'] = esmi_frag['n_ring'] + nRing_base

        # delete '*' at the insertion location
        esmi_base = esmi_base.drop(idx_base).reset_index(drop=True)

        # combine base with the fragment
        ext_smi = pd.concat([esmi_base.iloc[:idx_base], esmi_frag, esmi_base.iloc[idx_base:]]).reset_index(drop=True)
        new_pd_row = {'esmi': '!', 'n_br': 0, 'n_ring': 0, 'substr': ['!']}
        ext_smi.append(new_pd_row, ignore_index=True)

        return ngram.esmi2smi(ext_smi)
    
    def __call__(self, samples, beta, *, size=None, p_frag=0.1, yield_lpf=False):
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
        p_frag: float
            Probability of performing a fragment based mutation
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
                if np.random.random() < p_frag:
                    frag_list = self.fragmenting(re_samples)
                    samples = [self.combine_fragments(np.random.choice(frag_list), np.random.choice(frag_list)) for _ in range(size)]
                else:
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
                
