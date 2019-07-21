#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ['collate_pool', 'CrystalGraphDataset', 'check_cuda', 'to_tensor', 'T_Data']


def check_cuda(cuda: Union[bool, str, torch.device]) -> torch.device:
    if isinstance(cuda, bool):
        if cuda:
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                raise RuntimeError('could not use CUDA on this machine')
        else:
            return torch.device('cpu')

    if isinstance(cuda, str):
        if 'cuda' in cuda:
            if torch.cuda.is_available():
                return torch.device(cuda)
            else:
                raise RuntimeError('could not use CUDA on this machine')
        elif 'cpu' in cuda:
            return torch.device('cpu')
        else:
            raise RuntimeError('wrong device identifier'
                               'see also: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device')

    if isinstance(cuda, torch.device):
        return cuda


def collate_pool(dataset_list):
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
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target) \
            in enumerate(dataset_list):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        base_idx += n_i
    return (torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx), \
           torch.stack(batch_target, dim=0)


T_Data = Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor]


def to_tensor(data: T_Data, *, unsqueeze: int = None) -> torch.Tensor:
    """
    Convert data to :class:`torch.Tensor`.

    Parameters
    ----------
    data: Union[pd.DataFrame, pd.Series, np.ndarray, list, tuple]
        Input dataset
    unsqueeze: int
        Returns new tensor with a dimension of size one inserted at the specified position.
        See Also: https://pytorch.org/docs/stable/torch.html#torch.unsqueeze

    Returns
    -------
    torch.Tensor

    """

    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if not isinstance(data, torch.Tensor):
        raise RuntimeError(
            'input must be pd.DataFrame, pd.Series, np.ndarray, or torch.Tensor but got %s' % data.__class__)

    if unsqueeze is not None:
        return data.unsqueeze(unsqueeze)
    return data


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

    def __init__(self, crystal_features, targets):
        self.targets = targets
        if isinstance(crystal_features, pd.DataFrame):
            crystal_features = crystal_features.values
        self.crystal_features = crystal_features

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        features, target = self.crystal_features[idx], self.targets[idx]
        return (features[0], features[1][0], features[1][1]), torch.Tensor([target])
