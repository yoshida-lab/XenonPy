#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch

from xenonpy.model.training.extension.base import BaseExtension

__all__ = ['TensorConverter']

T_Data = Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor]


class TensorConverter(BaseExtension):

    def __init__(self, ):
        super().__init__()

    def input_proc(self,
                   x_in: Union[T_Data, Tuple[T_Data]],
                   y_in=None,
                   train: bool = True, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert data to :class:`torch.Tensor`.

        Parameters
        ----------
        x_in: Union[T_Data, Tuple[T_Data]]
            Dataset as model inputs.
        y_in: Union[T_Data, Tuple[T_Data]]
            Dataset as targets.
        train: bool
            Specify whether the model in the training mode.

        Returns
        -------
        Union[Any, Tuple[Any, Any]]

        """

        def _convert(t):
            if t is None:
                return t
            if isinstance(t, (pd.DataFrame, pd.Series)):
                t = t.values
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t)
            if not isinstance(t, torch.Tensor):
                raise RuntimeError(
                    'input must be pd.DataFrame, pd.Series, np.ndarray, or torch.Tensor but got %s' % t.__class__)

            if len(t.size()) == 1:
                t = t.unsqueeze(-1)
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

    def output_proc(self,
                    y_pred: Union[torch.Tensor, Tuple[torch.Tensor]],
                    train: bool = True, **kwargs):
        """
        Convert :class:`torch.Tensor` to :class:`numpy.ndarray`.

        Parameters
        ----------
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor]]
        train: bool
            Specify whether the model in the training mode.
        kwargs

        """
        if not train:
            if isinstance(y_pred, tuple):
                y = tuple([t.detach().cpu().numpy() for t in y_pred])
            else:
                y = y_pred.detach().cpu().numpy()
            return y
        else:
            return y_pred
