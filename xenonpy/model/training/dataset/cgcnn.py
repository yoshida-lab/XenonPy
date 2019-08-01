#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ['CrystalGraphDataset']


class CrystalGraphDataset(Dataset):
    """

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int or None
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (1, )
    cif_id: str or int
    """

    def __init__(self, crystal_features: Union[pd.DataFrame, np.ndarray],
                 targets: Union[pd.DataFrame, np.ndarray] = None):
        if isinstance(crystal_features, pd.DataFrame):
            crystal_features = crystal_features.values
        if not isinstance(crystal_features, np.ndarray):
            raise RuntimeError('<crystal_features> must be pd.DataFrame or np.ndarray')
        self.crystal_features = crystal_features

        if targets is not None:
            if isinstance(targets, pd.DataFrame):
                targets = targets.values
            if not isinstance(targets, np.ndarray):
                raise RuntimeError('<targets> must be pd.DataFrame, pd.Series or np.ndarray')
        self.targets = targets

    def __len__(self):
        return self.crystal_features.shape[0]

    def __getitem__(self, idx):
        features = self.crystal_features[idx]
        if self.targets is not None:
            target = self.targets[idx]
            return (features[0], features[1], features[2]), torch.Tensor(target)
        return features[0], features[1], features[2]

    @staticmethod
    def collate_fn(dataset_list):
        """
        Collate a list of data and return a batch for predicting crystal
        properties.

        Parameters
        ----------

        dataset_list: list of tuples for each data point.
          (atom_fea, nbr_fea, nbr_fea_idx, target)

          atom_fea: torch.Tensor shape (n_i, atom_fea_len)
          nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
          nbr_fea_idx: torch.LongTensor shape (n_i, M)
          target: torch.Tensor shape (1, )
          cif_id: str or int

        Returns
        -------
        N = sum(n_i); N0 = sum(i)

        batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
          Atom features from atom type
        batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        batch_nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          Mapping from the crystal idx to atom idx
        target: torch.Tensor shape (N, 1)
          Target value for prediction
        batch_cif_ids: list
        """

        def _batch():
            batch_atom_fea.append(atom_fea)
            batch_nbr_fea.append(nbr_fea)
            batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
            new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
            crystal_atom_idx.append(new_idx)

        batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
        crystal_atom_idx, batch_target = [], []
        base_idx = 0

        if len(dataset_list[0]) == 2:
            for i, ((atom_fea, nbr_fea, nbr_fea_idx), target) in enumerate(dataset_list):
                n_i = atom_fea.shape[0]  # number of atoms for this crystal
                _batch()
                base_idx += n_i
                batch_target.append(target)
            return (torch.cat(batch_atom_fea, dim=0),
                    torch.cat(batch_nbr_fea, dim=0),
                    torch.cat(batch_nbr_fea_idx, dim=0),
                    crystal_atom_idx), torch.stack(batch_target, dim=0)

        else:
            for i, (atom_fea, nbr_fea, nbr_fea_idx) in enumerate(dataset_list):
                n_i = atom_fea.shape[0]  # number of atoms for this crystal
                _batch()
                base_idx += n_i
            return (torch.cat(batch_atom_fea, dim=0),
                    torch.cat(batch_nbr_fea, dim=0),
                    torch.cat(batch_nbr_fea_idx, dim=0),
                    crystal_atom_idx)
