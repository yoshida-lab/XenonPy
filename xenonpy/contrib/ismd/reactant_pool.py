#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
# -*- coding: utf-8 -*-

from xenonpy.inverse.base import BaseProposal, ProposalError
import random
import pandas as pd


class ReactantNotInPoolError(ProposalError):

    def __init__(self, r_id):
        super().__init__("reactant id {} is not in the reactant pool".format(r_id))


class NotSquareError(ProposalError):

    def __init__(self, n_row=0, n_col=0):
        super().__init__("dataframe of shape {} * {} is not square".format(n_row, n_col))


class SimPoolnotmatchError(ProposalError):

    def __init__(self, n_pool=0, n_sim=0):
        super().__init__(
            "reactant pool with length {} dose not match the size of similarity matrix of size {} * {}"
            .format(n_pool, n_sim, n_sim))


class NoSampleError(ProposalError):

    def __init__(self):
        super().__init__("sample is empty")


class ReactantPool(BaseProposal):

    def __init__(self,
                 *,
                 pool_df=None,
                 sim_df=None,
                 reactor=None,
                 pool_smiles_col='SMILES',
                 sim_id_in_pool=None,
                 sample_reactant_idx_col='reactant_idx',
                 sample_reactant_smiles_col='reactant_smiles',
                 sample_product_smiles_col='product_smiles',
                 splitter='.'):
        """
        A module consists the reactant pool for proposal

        Parameters
        ----------
        pool_df: [pandas.DataFrame]
            a dataframe contains the all usable reactants,
        sim_df: [pandas.DataFrame]
            a matrix of similarity between each pair of reactant in pool_df, size len(pool_df)*len(pool_df).
        reactor: [xenonpy.contrib.ismd.Reactor]
            reaction prediction model.
        pool_smiles_col: [str]
            the column name of reactant SMILES in the pool_df,
        sim_id_in_pool: [str]
            the column name of id in pool_df corresponding to sim_df, if None, use index.
        sample_reactant_idx_col: [str]
            the column name of reactant id in sample dataframe.
        #sample_reactant_idx_old_col: [str]
            the column name of old reactant id in sample dataframe, copied from sample_reactant_idx_col before proposal.
        sample_reactant_smiles_col: [str]
            the column name of reactant SMILES in sample dataframe
        sample_product_smiles_col: [str]
            the column name of product SMILES in sample dataframe
        splitter: [str]
            string used for concatenating reactant in a reactant set.
        """
        if sim_df.shape[0] != sim_df.shape[1]:
            raise NotSquareError(sim_df.shape[0], sim_df.shape[1])

        if pool_df.shape[0] != sim_df.shape[0]:
            raise SimPoolnotmatchError(pool_df.shape[0], sim_df.shape[0])

        if sim_id_in_pool is not None:
            self._pool_df = pool_df.set_index(sim_id_in_pool)
        else:
            self._pool_df = pool_df
        self._sim_df = sim_df
        self._reactor = reactor
        self._pool_smiles_col = pool_smiles_col
        self._sample_reactant_idx_col = sample_reactant_idx_col
        self._sample_reactant_smiles_col = sample_reactant_smiles_col
        self._sample_product_smiles_col = sample_product_smiles_col
        self._splitter = splitter

    def single_index2reactant(self, reactant) -> str:
        """
        convert a list of index to string concatenated by the splitter.

        Parameters
        ----------
        reactant: [list]
            list of reactant id (e.g. [45011, 29392])

        Returns
        -------
            reactant_smiles: [str]
                reactant SMILES concatenated by the splitter (e.g. CC(C)Br.COc1ccc(C=O)cc1O)
        """
        r_list = [self._pool_df.iloc[r][self._pool_smiles_col] for r in reactant]
        return self._splitter.join(r_list)

    def index2sim(self, r_idx_old):
        """
        substitute the reactant id by a similar one.

        Parameters
        ----------
        r_idx_old: [int]
            reactant id to be substituted (e.g. 29392)

        Returns
        -------
            r_idx_new: [int]
                an id of reactant similar to r_idx_old (e.g. 9980)
        """
        if r_idx_old not in self._pool_df.index:
            raise ReactantNotInPoolError(r_idx_old)
        sim_col = self._sim_df[[r_idx_old]].drop(r_idx_old)
        sim_col = sim_col.loc[sim_col[r_idx_old] != 0]
        if len(sim_col) > 0:
            r_idx_new = random.choices(population=sim_col.index, weights=sim_col[r_idx_old], k=1)[0]
        else:
            r_idx_new = random.choices(population=self._sim_df[[r_idx_old]].drop(r_idx_old).index,
                                       k=1)[0]
        return r_idx_new

    def single_proposal(self, reactant):
        """
        substitute a randomly selected reactant id in reactant by a similar one.

        Parameters
        ----------
        reactant: [list]
            list of id (e.g. [45011, 29392])

        Returns
        -------
        r_idx_new: [list]
            list of id (e.g. [45011, 9980])
        """
        modify_idx = random.choice(list(range(len(reactant))))
        r_idx_old = reactant[modify_idx]
        r_idx_new = self.index2sim(r_idx_old)
        new_reactant = reactant[:]
        new_reactant[modify_idx] = r_idx_new
        return new_reactant

    def proposal(self, sample_df):
        """
        1, perform single_propopsal to all of the reactant list,
        2, convert reactant id to reactant SMILES
        3, perform reactant prediction

        Parameters
        ----------
        sample_df: [pandas.DataFrame]
            sample dataframe

        Returns
        -------
        sample_df: [pandas.DataFrame]
            modified sample dataframe 
        """
        if len(sample_df) == 0:
            raise NoSampleError()
        new_sample_df = pd.DataFrame(columns=sample_df.columns)
        old_list = [list(r) for r in sample_df[self._sample_reactant_idx_col]]
        new_sample_df[self._sample_reactant_idx_col] = [
            self.single_proposal(reactant) for reactant in old_list
        ]
        new_sample_df[self._sample_reactant_smiles_col] = [
            self.single_index2reactant(id_str)
            for id_str in new_sample_df[self._sample_reactant_idx_col]
        ]
        new_sample_df[self._sample_product_smiles_col] = self._reactor.react(
            new_sample_df[self._sample_reactant_smiles_col])
        return new_sample_df
