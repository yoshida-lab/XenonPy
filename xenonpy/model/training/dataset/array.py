#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

__all__ = ['ArrayDataset']


class ArrayDataset(TensorDataset):

    def __init__(self, *array: Union[np.ndarray, pd.DataFrame, pd.Series, torch.Tensor], dtype=None):
        if dtype is None:
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype

        array = [self._convert(data) for data in array]
        super().__init__(*array)

    def _convert(self, data):
        if isinstance(data, torch.Tensor):
            return data
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if not isinstance(data, torch.Tensor):
            raise RuntimeError(
                'input must be pd.DataFrame, pd.Series, np.ndarray, or torch.Tensor but got %s' % data.__class__)

        if len(data.size()) == 1:
            data = data.unsqueeze(-1)

        return data.to(self.dtype)
