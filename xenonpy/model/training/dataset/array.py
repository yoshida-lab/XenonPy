#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from typing import Sequence, Union

__all__ = ['ArrayDataset']


class ArrayDataset(TensorDataset):
    def __init__(self,
                 *array: Union[np.ndarray, pd.DataFrame, pd.Series,
                               torch.Tensor],
                 dtypes: Union[None, Sequence[torch.dtype]] = None):
        if dtypes is None:
            dtypes = [torch.get_default_dtype()] * len(array)
        if len(dtypes) != len(array):
            raise ValueError('length of dtypes not equal to length of array')

        array = [
            self._convert(data, dtype) for data, dtype in zip(array, dtypes)
        ]
        super().__init__(*array)

    @staticmethod
    def _convert(data, dtype):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if not isinstance(data, torch.Tensor):
            raise RuntimeError(
                'input must be pd.DataFrame, pd.Series, np.ndarray, or torch.Tensor but got %s'
                % data.__class__)

        return data.to(dtype)
