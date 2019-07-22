#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union

import numpy as np
import pandas as pd
import torch

__all__ = ['check_cuda', 'to_tensor', 'T_Data']


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
