#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch

from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension

__all__ = ['TensorConverter']

T_Data = Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor]


class TensorConverter(BaseExtension):

    def __init__(self, dtype=None, empty_cache: bool = False):
        self.empty_cache = empty_cache
        if dtype is None:
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype

    def input_proc(self, x_in, y_in, trainer: Trainer) -> Tuple[torch.Tensor, torch.Tensor]:
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
            if t is None:
                return t
            if isinstance(t, (tuple, list)):
                return tuple([_convert(t_) for t_ in t])

            # if tensor, do nothing
            if isinstance(t, torch.Tensor):
                return t.to(trainer.device, non_blocking=trainer.non_blocking)
            # if pandas, turn to numpy
            if isinstance(t, (pd.DataFrame, pd.Series)):
                t = t.values
            # if numpy, turn to tensor
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t).to(self.dtype)
            # return others
            if not isinstance(t, torch.Tensor):
                return t
            # reshape (1,) to (-1, 1)
            if len(t.size()) == 1:
                t = t.unsqueeze(-1)
            return t.to(trainer.device, non_blocking=trainer.non_blocking)

        return _convert(x_in), _convert(y_in)

    def step_forward(self):
        if self.empty_cache:
            torch.cuda.empty_cache()

    def output_proc(self,
                    y_pred: Union[torch.Tensor, Tuple[torch.Tensor]],
                    y_true: Union[torch.Tensor, Tuple[torch.Tensor], None],
                    training: bool,
                    ):
        """
        Convert :class:`torch.Tensor` to :class:`numpy.ndarray`.

        Parameters
        ----------
        y_pred: Union[torch.Tensor, Tuple[torch.Tensor]]
        y_true : Union[torch.Tensor, Tuple[torch.Tensor]]
        training: bool
            Specify whether the model in the training mode.

        """

        def _convert(y_):
            if y_ is None:
                return y_
            if isinstance(y_, tuple):
                y_ = tuple([t.detach().cpu().numpy() for t in y_])
            else:
                y_ = y_.detach().cpu().numpy()
            return y_

        if not training:
            return _convert(y_pred), _convert(y_true)
        else:
            return y_pred, y_true
