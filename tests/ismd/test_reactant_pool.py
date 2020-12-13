#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 00:39:20 2020

@author: qiz
"""

import pytest
import pandas as pd
from xenonpy.contrib.ismd import ReactantPool
from xenonpy.contrib.ismd.reactant_pool import NotSquareError, SimPoolnotmatchError, ReactantNotInPoolError, NoSampleError


def test_NotSquareError():

    reactant_pool = pd.DataFrame({
        "id": [0, 1, 2, 3, 4],
        "SMILES": [
            "O=C(Cl)Oc1ccc(Cc2ccc(C(F)(F)F)cc2)cc1", "CC(NC(=O)OCc1ccccc1)C(C)NC(=O)c1ccccc1O", "OC[C@H]1NCC[C@@H]1O",
            "C#CCCN1C(=O)c2ccccc2C1=O", "CC(=O)OCCS(=O)(=O)Cl"
        ]
    })
    sim_df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6]], columns=['0', '1', '2'])
    with pytest.raises(NotSquareError):
        ReactantPool(pool_df=reactant_pool, sim_df=sim_df, pool_smiles_col='SMILES')


def test_SimPoolnotmatchError():

    reactant_pool = pd.DataFrame({
        "id": [0, 1, 2, 3, 4],
        "SMILES": [
            "O=C(Cl)Oc1ccc(Cc2ccc(C(F)(F)F)cc2)cc1", "CC(NC(=O)OCc1ccccc1)C(C)NC(=O)c1ccccc1O", "OC[C@H]1NCC[C@@H]1O",
            "C#CCCN1C(=O)c2ccccc2C1=O", "CC(=O)OCCS(=O)(=O)Cl"
        ]
    })
    sim_df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['0', '1', '2'])
    with pytest.raises(SimPoolnotmatchError):
        ReactantPool(pool_df=reactant_pool, sim_df=sim_df, pool_smiles_col='SMILES')


def test_ReactantNotInPoolError():
    reactant_pool = pd.DataFrame({
        "id": [0, 1, 2],
        "SMILES": [
            "O=C(Cl)Oc1ccc(Cc2ccc(C(F)(F)F)cc2)cc1", "CC(NC(=O)OCc1ccccc1)C(C)NC(=O)c1ccccc1O", "OC[C@H]1NCC[C@@H]1O"
        ]
    })
    sim_df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['0', '1', '2'])
    reactant_pool = ReactantPool(pool_df=reactant_pool, sim_df=sim_df, pool_smiles_col='SMILES')
    with pytest.raises(ReactantNotInPoolError):
        reactant_pool.index2sim(5)


def test_NoSampleError():
    reactant_pool = pd.DataFrame({
        "id": [0, 1, 2],
        "SMILES": [
            "O=C(Cl)Oc1ccc(Cc2ccc(C(F)(F)F)cc2)cc1", "CC(NC(=O)OCc1ccccc1)C(C)NC(=O)c1ccccc1O", "OC[C@H]1NCC[C@@H]1O"
        ]
    })
    sim_df = pd.DataFrame(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=['0', '1', '2'])
    reactant_pool = ReactantPool(pool_df=reactant_pool, sim_df=sim_df, pool_smiles_col='SMILES')
    init_samples = pd.DataFrame({'reactant_idx': [], 'reactant_smiles': [], 'product_smiles': []})
    with pytest.raises(NoSampleError):
        reactant_pool.proposal(init_samples)


class Reactor():

    def __init__(self):
        """
        A chemical reaction prediction model
        ----------
        Parameters:
            model : A molecular transformer model for reaction prediction
        """

    def react(self, reactant_list) -> list:
        """
        Tokenize a SMILES molecule or reaction
        ----------
        Parameters:
            reactant_list : list of reactant 
        Returns:
            product_list: all_predictions is a list of `batch_size` lists of `n_best` predictions
        """
        product_list = [r.replace('.', '') for r in reactant_list]
        return product_list


def test_poolAssignment():

    reactant_pool = pd.DataFrame({"id": [0, 1, 2], "SMILES": ["C", "O", "N"]})
    sim_df = pd.DataFrame(data=[[1, 0.7, 0.2], [0.7, 1, 0], [0.2, 0, 1]], columns=[0, 1, 2])
    fake_reactor = Reactor()
    test_pool = ReactantPool(pool_df=reactant_pool, sim_df=sim_df, reactor=fake_reactor, pool_smiles_col='SMILES')
    assert test_pool._pool_smiles_col in test_pool._pool_df


def test_sampleAssignment():

    reactant_pool = pd.DataFrame({"id": [0, 1, 2], "SMILES": ["C", "O", "N"]})
    sim_df = pd.DataFrame(data=[[1, 0.7, 0.2], [0.7, 1, 0], [0.2, 0, 1]], columns=[0, 1, 2])
    fake_reactor = Reactor()
    test_pool = ReactantPool(pool_df=reactant_pool,
                             sim_df=sim_df,
                             reactor=fake_reactor,
                             pool_smiles_col='SMILES',
                             sample_reactant_idx_col='reactant_idx',
                             sample_reactant_smiles_col='reactant_smiles',
                             sample_product_smiles_col='product_smiles')
    sample_df = pd.DataFrame({
        'reactant_idx': [[2, 1], [0, 1], [1, 0], [0, 2]],
        'reactant_smiles': ["N.O", "C.O", "O.C", "C.N"],
        'product_smiles': ["NO", "CO", "OC", "CN"]
    })

    assert isinstance(test_pool._pool_df, pd.DataFrame)
    assert isinstance(test_pool._sim_df, pd.DataFrame)
    assert test_pool._sample_reactant_idx_col in sample_df
    assert test_pool._sample_reactant_smiles_col in sample_df
    assert test_pool._sample_product_smiles_col in sample_df


def test_singleProposal():
    reactant_pool = pd.DataFrame({"id": [0, 1, 2], "SMILES": ["C", "O", "N"]})
    sim_df = pd.DataFrame(data=[[1, 0.7, 0.2], [0.7, 1, 0], [0.2, 0, 1]], columns=[0, 1, 2])
    fake_reactor = Reactor()
    test_pool = ReactantPool(pool_df=reactant_pool,
                             sim_df=sim_df,
                             reactor=fake_reactor,
                             pool_smiles_col='SMILES',
                             sample_reactant_idx_col='reactant_idx',
                             sample_reactant_smiles_col='reactant_smiles',
                             sample_product_smiles_col='product_smiles')
    sample_df = pd.DataFrame({
        'reactant_idx': [[2, 1], [0], [1, 0, 2], [0, 2]],
        'reactant_smiles': ["N.O", "C", "O.C.N", "C.N"],
        'product_smiles': ["NO", "C", "OCN", "CN"]
    })

    old_list = [[2, 1], [0], [1, 0, 2], [0, 2]]
    new_list = [test_pool.single_proposal(reactant) for reactant in sample_df[test_pool._sample_reactant_idx_col]]

    assert len(old_list) == len(new_list)
    assert all([len(o) == len(n) for o, n in zip(old_list, new_list)])

    def list_only_one_diff(list1, list2):
        return len(set(list1) - set(list2)) == 1

    assert all([list_only_one_diff(o, n) for o, n in zip(old_list, new_list)])


def test_reactant_id2smiles():
    reactant_pool = pd.DataFrame({"id": [0, 1, 2], "SMILES": ["C", "O", "N"]})
    sim_df = pd.DataFrame(data=[[1, 0.7, 0.2], [0.7, 1, 0], [0.2, 0, 1]], columns=[0, 1, 2])
    fake_reactor = Reactor()
    test_pool = ReactantPool(pool_df=reactant_pool,
                             sim_df=sim_df,
                             reactor=fake_reactor,
                             pool_smiles_col='SMILES',
                             sample_reactant_idx_col='reactant_idx',
                             sample_reactant_smiles_col='reactant_smiles',
                             sample_product_smiles_col='product_smiles')
    sample_df = pd.DataFrame({
        'reactant_idx': [[2, 1], [0], [1, 0, 2], [0, 2]],
        'reactant_smiles': ["", "", "", ""],
        'product_smiles': ["", "", "", ""]
    })
    sample_df[test_pool._sample_reactant_smiles_col] = [
        test_pool.single_index2reactant(id_str) for id_str in sample_df[test_pool._sample_reactant_idx_col]
    ]

    assert len(sample_df[test_pool._sample_reactant_idx_col]) == len(sample_df[test_pool._sample_reactant_smiles_col])
    assert all([
        len(idx) == len(smi.split(test_pool._splitter)) for idx, smi in zip(
            sample_df[test_pool._sample_reactant_idx_col], sample_df[test_pool._sample_reactant_smiles_col])
    ])


def test_reactant2product():
    reactant_pool = pd.DataFrame({"id": [0, 1, 2], "SMILES": ["C", "O", "N"]})
    sim_df = pd.DataFrame(data=[[1, 0.7, 0.2], [0.7, 1, 0], [0.2, 0, 1]], columns=[0, 1, 2])
    fake_reactor = Reactor()
    test_pool = ReactantPool(pool_df=reactant_pool,
                             sim_df=sim_df,
                             reactor=fake_reactor,
                             pool_smiles_col='SMILES',
                             sample_reactant_idx_col='reactant_idx',
                             sample_reactant_smiles_col='reactant_smiles',
                             sample_product_smiles_col='product_smiles')
    sample_df = pd.DataFrame({
        'reactant_idx': [[2, 1], [0], [1, 0, 2], [0, 2]],
        'reactant_smiles': ["N.O", "C", "O.C.N", "C.N"],
        'product_smiles': ["", "", "", ""]
    })
    sample_df[test_pool._sample_product_smiles_col] = test_pool._reactor.react(
        sample_df[test_pool._sample_reactant_smiles_col])

    assert len(sample_df[test_pool._sample_product_smiles_col]) == len(sample_df[test_pool._sample_reactant_smiles_col])


def test_proposal():
    reactant_pool = pd.DataFrame({"id": [0, 1, 2], "SMILES": ["C", "O", "N"]})
    sim_df = pd.DataFrame(data=[[1, 0.7, 0.2], [0.7, 1, 0], [0.2, 0, 1]], columns=[0, 1, 2])
    fake_reactor = Reactor()
    test_pool = ReactantPool(pool_df=reactant_pool,
                             sim_df=sim_df,
                             reactor=fake_reactor,
                             pool_smiles_col='SMILES',
                             sample_reactant_idx_col='reactant_idx',
                             sample_reactant_smiles_col='reactant_smiles',
                             sample_product_smiles_col='product_smiles')
    old_df = pd.DataFrame({
        'reactant_idx': [[2, 1], [0], [1, 0, 2], [0, 2]],
        'reactant_smiles': ["", "", "", ""],
        'product_smiles': ["", "", "", ""]
    })
    new_df = test_pool.proposal(old_df)

    assert len(old_df) == len(new_df)
    assert list(old_df) == list(new_df)
    assert all([
        oidx != nidx
        for oidx, nidx in zip(old_df[test_pool._sample_reactant_idx_col], new_df[test_pool._sample_reactant_idx_col])
    ])
    assert not new_df.isnull().values.any()
