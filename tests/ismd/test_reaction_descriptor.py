#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:39:20 2020

@author: qiz
"""

import pytest
import pandas as pd
from xenonpy.contrib.ismd.reaction_descriptor import NoSample2FeatureError
from xenonpy.descriptor import Fingerprints
from xenonpy.contrib.ismd import ReactionDescriptor


def test_reactantpool_sim_shape():

    mol_featurizer = Fingerprints(featurizers=['ECFP', 'MACCS'], input_type='smiles', on_errors='nan')
    rd = ReactionDescriptor(descriptor_calculator=mol_featurizer,
                            return_type='df',
                            target_col="product_smiles",
                            on_errors='nan')
    init_samples = pd.DataFrame({'reactant_idx': [], 'reactant_smiles': [], 'product_smiles': []})
    with pytest.raises(NoSample2FeatureError):
        rd.featurize(init_samples)
