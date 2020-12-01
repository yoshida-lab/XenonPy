#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
# -*- coding: utf-8 -*-

import numpy as np
from xenonpy.inverse.base import BaseProposal, ProposalError
import random


class IndexOutError(ProposalError):

    def __init__(self, *, discarded_index="", pool_list=[]):
        self.discarded_index = discarded_index
        self.modified_index = str(random.choices(pool_list, k=1)[0])

        super().__init__('index: {} if out of reactant pool bound: {}'.format(discarded_index, len(self.pool_list)))


class InvaidIndexError(ProposalError):

    def __init__(self, *, discarded_index="", pool_list=[]):
        self.discarded_index = discarded_index
        self.modified_index = str(random.choices(pool_list, k=1)[0])

        super().__init__('input: {} is not a valid index'.format(discarded_index))

class NotSquareError(ProposalError):
    
    def __init__(self, n_row=0, n_col=0):
        super().__init__("dataframe of shape {} * {} is not square".format(n_row, n_col))

class ReactantPool(BaseProposal):

    def __init__(self,
                 *,
                 pool_df=None,
                 sim_df=None,
                 reactor=None,
                 pool_smiles_col='SMILES',
                 sim_id_in_pool=None,
                 sample_reactant_idx_col='reactant_idx',
                 #sample_reactant_idx_old_col='reactant_idx_old',
                 sample_reactant_smiles_col='reactant_smiles',
                 sample_product_smiles_col='product_smiles',
                 splitter='.'):
        """
        A module consists the reactant pool for proposal
        ----------
        Parameters:
            pool_df : a dataframe contains the all usable reactants,
            sim_df : a matrix of similarity between each pair of reactant in pool_df, size len(pool_df)*len(pool_df).
            reactor : reaction prediction model.
            pool_smiles_col : the column name of reactant SMILES in the pool_df,
            sim_id_in_pool : the column name of id in pool_df corresponding to sim_df, if None, use index.
            sample_reactant_idx_col : the column name of reactant id in sample dataframe.
            #sample_reactant_idx_old_col : the column name of old reactant id in sample dataframe, copied from sample_reactant_idx_col before proposal.
            sample_reactant_smiles_col : the column name of reactant SMILES in sample dataframe
            sample_product_smiles_col : the column name of product SMILES in sample dataframe
            splitter : string used for concatenating reactant in a reactant set.
        """
        if len(sim_df) != len(sim_df.columns):
            raise NotSquareError(len(sim_df), len(sim_df.columns))
            
        if sim_id_in_pool is not None:
            self._pool_df = pool_df.set_index(sim_id_in_pool)
        else:
            self._pool_df = pool_df
        self._sim_df = sim_df
        self._reactor = reactor
        self._pool_smiles_col = pool_smiles_col
        #self._sim_id_in_pool = sim_id_in_pool
        self._sample_reactant_idx_col = sample_reactant_idx_col
        #self._sample_reactant_idx_old_col = sample_reactant_idx_old_col
        self._sample_reactant_smiles_col = sample_reactant_smiles_col
        self._sample_product_smiles_col = sample_product_smiles_col
        self._splitter = splitter

    def on_errors(self, error):
        """

        Parameters
        ----------
        error: ProposalError
            Error object.
        Returns
        -------

        """
        if isinstance(error, IndexOutError):
            return error.modified_index
        if isinstance(error, InvaidIndexError):
            return error.modified_index

    def single_index2reactant(self, reactant) -> str:
        """
        convert a list of index to string concatenated by the splitter.
        ----------
        Parameters:
            reactant : list of reactant id (e.g. [45011, 29392])
        Returns:
            reactant_smiles : reactant SMILES concatenated by the splitter (e.g. CC(C)Br.COc1ccc(C=O)cc1O)
        """
        r_list = [self._pool_df.iloc[r][self._pool_smiles_col] for r in reactant]
        reactant_smiles = self._splitter.join(r_list)
        return reactant_smiles

    def index2sim(self, r_idx_old):
        """
        substitute the reactant id by a similar one.
        ----------
        Parameters:
            r_idx_old : reactant id to be substituted (e.g. 29392)
        Returns:
            r_idx_new : an id of reactant similar to r_idx_old (e.g. 9980)
        """
        sim_col = self._sim_df[[r_idx_old]].drop(r_idx_old)
        sim_col = sim_col.loc[sim_col[r_idx_old] != 0]
        if len(sim_col) > 0:
            r_idx_new = random.choices(population=sim_col.index, weights=sim_col[r_idx_old], k=1)[0]
        else:
            r_idx_new = random.choices(population=self._sim_df[[r_idx_old]].drop(r_idx_old).index, k=1)[0]
        return r_idx_new

    def single_propopsal(self, reactant):
        """
        substitute a randomly selected reactant id in reactant by a similar one.
        ----------
        Parameters:
            reactant : list of id (e.g. [45011, 29392])
        Returns:
            r_idx_new : list of id (e.g. [45011, 9980])
        """
        modify_idx = random.choice(list(range(len(reactant))))
        r_idx_old = reactant[modify_idx]
        r_idx_new = self.index2sim(r_idx_old)
        # =============================================================================
        #         try:
        #             if not discarded_index.isnumeric():
        #                 raise InvaidIndexError(discarded_index, self.pool[self.index_column].tolist())
        #             if int(discarded_index) >= len(self.pool):
        #                 raise IndexOutError(discarded_index, self.pool[self.index_column].tolist())
        #             reactant_idx_new = self.index2sim(reactant_idx_old)
        #         except ProposalError as e:
        #             modified_index = self.on_errors(e)
        #
        #         except Exception as e:
        #             raise e
        #
        # =============================================================================
        reactant[modify_idx] = r_idx_new
        return reactant

    def proposal(self, sample_df):
        """
        1, perform single_propopsal to all of the reactant list,
        2, convert reactant id to reactant SMILES
        3, perform reactant prediction
        ----------
        Parameters:
            sample_df : sample dataframe
        Returns:
            sample_df : sample dataframe 
        """
        #sample_df[self._sample_reactant_idx_old_col] = sample_df[self._sample_reactant_idx_col]
        old_list = [list(r) for r in sample_df[self._sample_reactant_idx_col]]  #ugly
        sample_df[self._sample_reactant_idx_col] = [self.single_propopsal(reactant) for reactant in old_list]
        sample_df[self._sample_reactant_smiles_col] = [
            self.single_index2reactant(id_str) for id_str in sample_df[self._sample_reactant_idx_col]
        ]
        sample_df[self._sample_product_smiles_col] = self._reactor.react(sample_df[self._sample_reactant_smiles_col])
        return sample_df
