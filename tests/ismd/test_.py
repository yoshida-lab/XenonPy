#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:39:20 2020

@author: qiz
"""

import pytest
import pandas as pd
from xenonpy.contrib.ismd import ReactantPool
from xenonpy.contrib.ismd.reactant_pool import ColNameError


@pytest.fixture
def input_value():
   input = 39
   return input

def test_divisible_by_3(input_value):
   assert input_value % 3 == 0

def test_divisible_by_6(input_value):
   assert input_value % 6 == 0

def test_reactantpool_colname():

    reactant_pool = pd.DataFrame({"id":[0,1,2,3,4], "SMILES":["O=C(Cl)Oc1ccc(Cc2ccc(C(F)(F)F)cc2)cc1","CC(NC(=O)OCc1ccccc1)C(C)NC(=O)c1ccccc1O","OC[C@H]1NCC[C@@H]1O",
                                                              "C#CCCN1C(=O)c2ccccc2C1=O","CC(=O)OCCS(=O)(=O)Cl"]})
    sim_df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=['0', '1', '2'])
    with pytest.raises(ColNameError):
        ReactantPool(pool_df=reactant_pool, sim_df=sim_df, pool_smiles_col='wrong_SMILES')

