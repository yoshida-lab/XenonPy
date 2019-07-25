#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch

from xenonpy.model.training.base import BaseExtension

__all__ = ['TensorConverter']

T_Data = Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor]


class TensorConverter(BaseExtension):

    def __init__(self, dtype=None):
        if dtype is 'default':
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype

    def input_proc(self, x_in, y_in, **_) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert data to :class:`torch.Tensor`.

        Parameters
        ----------
        y_in
        x_in

        Returns
        -------
        Union[Any, Tuple[Any, Any]]

        """

        def _convert(t):
            if isinstance(t, (pd.DataFrame, pd.Series)):
                t = t.values
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t)
            if not isinstance(t, torch.Tensor):
                return t

            if len(t.size()) == 1:
                t = t.unsqueeze(-1)
            if self.dtype is not None:
                return t.to(self.dtype)
            return t

        if isinstance(x_in, tuple):
            x_in = tuple([_convert(t) for t in x_in])
        else:
            x_in = _convert(x_in)

        if isinstance(y_in, tuple):
            y_in = tuple([_convert(t) for t in y_in])
        else:
            y_in = _convert(y_in)

        return x_in, y_in

    def output_proc(self, y_pred: Union[torch.Tensor, Tuple[torch.Tensor]], *, training, **_):
        """
        Convert :class:`torch.Tensor` to :class:`numpy.ndarray`.

        Parameters
        ----------
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor]]
        train_mode: bool
            Specify whether the model in the training mode.
        kwargs

        """
        if not training:
            if isinstance(y_pred, tuple):
                y = tuple([t.detach().cpu().numpy() for t in y_pred])
            else:
                y = y_pred.detach().cpu().numpy()
            return y
        else:
            return y_pred
