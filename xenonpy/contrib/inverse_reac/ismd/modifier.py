#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 20:53:29 2020

@author: qi
"""

from xenonpy.inverse.base import BaseProposal, ProposalError
import random

class PoolSampler(BaseProposal):
    def __init__(self, *, reactant_pool=None):
        self.reactant_pool = reactant_pool;
    

    def proposal(self, reactants):
        
# =============================================================================
#         print(self.reactant_pool[0])
# =============================================================================
        """
        Propose new SMILES based on the given SMILES.

        Parameters
        ----------
        reactants: list of reactants
            Given reactants in SMILES format for modification.

        Returns
        -------
        new_reacts: list of reactants
            The proposed reactants in SMILES format from the given SMILES.
        """
        
        
        new_reacts = []
        for i, react in enumerate(reactants):
            react = react.split(".")
            react[random.randint(0,len(react)-1)] = random.sample(self.reactant_pool,1)[0]
            new_react = '.'.join(react)
            new_reacts.append(new_react)

        return new_reacts   








