#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:46:32 2019

@author: qizhang
"""
import numpy as np
from xenonpy.inverse.base import BaseProposal, ProposalError

class ReactantPool(BaseProposal):
    
    def __init__(self, pool_data, similarity_matrix, splitter='.'):
        self.pool = pool_data
        self.sim = similarity_matrix
        self.splitter = splitter
        
    def generate_index_dict(self):
        self.id_dict = {}
        for i in range(len(self.pool)):
            self.id_dict[self.pool[i]]=i
            
    def single_index2reactant(self, index_str):
        index_list = index_str.split(self.splitter)
        r_list = [self.pool[int(id)] for id in index_list]
        reactant = '.'.join(r_list)
        return reactant
    
    def index2reactant(self, index):
        reactant_list = [self.single_index2reactant(id_str) for id_str in index]
        return reactant_list
    
    def index2sim(self, index):
        index = int(index)
        sim_list = self.sim[index,:].nonzero()[1].tolist()
        if len(sim_list)<=1:
            sim_list = np.random.choice(len(self.pool), 100).tolist()
        if index in sim_list:
            sim_list.remove(index) 
        sim_index = np.random.choice(sim_list, 1)[0]
        sim_index = str(sim_index)
        return sim_index
    
    def single_propopsal(self, index_str):
        index_list = index_str.split(self.splitter)
        proposal_iloc = np.random.choice(len(index_list),1)[0]
        index_list[proposal_iloc] = self.index2sim(index_list[proposal_iloc])
        index_proposal = self.splitter.join(index_list)
        return index_proposal
        
    
    def proposal(self, reactant_index):
        """
        Propose new index based on the given index.

        Parameters
        ----------
        reactant_index: list of reactant index
            Given reactant index of reactant pool in for modification.

        Returns
        -------
        proposal_reactant_index: list of reactant index
            The proposed reactants in index format from the given index.
        """
        
        proposal_reactant_index = [self.single_propopsal(index_str) for index_str in reactant_index]

        return proposal_reactant_index