#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:39:20 2020

@author: qiz
"""

import pytest
import pandas as pd
from xenonpy.contrib.ismd.reaction_descriptor import NoSample2FeatureError, NoSampleColumnError
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


def test_NoSampleColumnError():
    mol_featurizer = Fingerprints(featurizers=['ECFP', 'MACCS'], input_type='smiles', on_errors='nan')
    rd = ReactionDescriptor(descriptor_calculator=mol_featurizer,
                            return_type='df',
                            target_col="product_smiles",
                            on_errors='nan')
    sample_df = pd.DataFrame({
        'reactant_idx': [[], [], []],
        'reactant_smiles': ["", "", ""],
        'fake_col': [
            "O=C(Cl)Oc1ccc(Cc2ccc(C(F)(F)F)cc2)cc1", "CC(NC(=O)OCc1ccccc1)C(C)NC(=O)c1ccccc1O", "OC[C@H]1NCC[C@@H]1O"
        ]
    })
    with pytest.raises(NoSampleColumnError):
        rd.featurize(sample_df)


def test_featurize():
    mol_featurizer = Fingerprints(featurizers=['ECFP', 'MACCS'], input_type='smiles', on_errors='nan')
    rd = ReactionDescriptor(descriptor_calculator=mol_featurizer,
                            return_type='df',
                            target_col="product_smiles",
                            on_errors='nan')
    sample_df = pd.DataFrame({
        'reactant_idx': [[], [], []],
        'reactant_smiles': ["", "", ""],
        'product_smiles': [
            "O=C(Cl)Oc1ccc(Cc2ccc(C(F)(F)F)cc2)cc1", "CC(NC(=O)OCc1ccccc1)C(C)NC(=O)c1ccccc1O", "OC[C@H]1NCC[C@@H]1O"
        ]
    })

    sample_fp = rd.featurize(sample_df)

    assert len(sample_fp) == len(sample_df)
    assert len(list(sample_fp)) == 2215
