#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:39:20 2020

@author: qiz
"""

import pytest
import pandas as pd
from xenonpy.contrib.ismd import ReactantPool
from xenonpy.contrib.ismd.reactant_pool import NotSquareError
from xenonpy.inverse.base import BaseProposal, ProposalError



def test_reactantpool_sim_shape():

    reactant_pool = pd.DataFrame({"id":[0,1,2,3,4], "SMILES":["O=C(Cl)Oc1ccc(Cc2ccc(C(F)(F)F)cc2)cc1",
                                                              "CC(NC(=O)OCc1ccccc1)C(C)NC(=O)c1ccccc1O",
                                                              "OC[C@H]1NCC[C@@H]1O",
                                                              "C#CCCN1C(=O)c2ccccc2C1=O",
                                                              "CC(=O)OCCS(=O)(=O)Cl"]})
    sim_df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=['0', '1', '2'])
    with pytest.raises(NotSquareError):
        ReactantPool(pool_df=reactant_pool, sim_df=sim_df, pool_smiles_col='wrong_SMILES')


