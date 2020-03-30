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

    def __init__(self,
                 x_dtype: Union[torch.dtype, Tuple[torch.dtype]] = None,
                 y_dtype: Union[torch.dtype, Tuple[torch.dtype]] = None,
                 empty_cache: bool = False,
                 classification: bool = False,
                 ):

        self.classification = classification
        self.empty_cache = empty_cache
        if x_dtype is None:
            self._x_dtype = torch.get_default_dtype()
        elif isinstance(x_dtype, Tuple):
            self._x_dtype = x_dtype
        else:
            self._x_dtype = x_dtype

        if y_dtype is None:
            self._y_dtype = torch.get_default_dtype()
        elif isinstance(y_dtype, Tuple):
            self._y_dtype = y_dtype
        else:
            self._y_dtype = y_dtype

    def _get_x_dtype(self, i=0):
        if isinstance(self._x_dtype, Tuple):
            return self._x_dtype[i]
        return self._x_dtype

    def _get_y_dtype(self, i=0):
        if isinstance(self._y_dtype, Tuple):
            return self._y_dtype[i]
        return self._y_dtype

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

        def _convert(t, dtype):
            if t is None:
                return t

            # if tensor, do nothing
            if isinstance(t, torch.Tensor):
                return t.to(trainer.device, non_blocking=trainer.non_blocking)
            # if pandas, turn to numpy
            if isinstance(t, (pd.DataFrame, pd.Series)):
                t = t.values
            # if numpy, turn to tensor
            if isinstance(t, np.ndarray):
                t = torch.from_numpy(t).to(dtype)
            # return others
            if not isinstance(t, torch.Tensor):
                return t
            # reshape (1,) to (-1, 1)
            if len(t.size()) == 1:
                t = t.unsqueeze(-1)
            return t.to(trainer.device, non_blocking=trainer.non_blocking)

        if isinstance(x_in, [tuple, list]):
            x_in = tuple([_convert(t, self._get_x_dtype(i)) for i, t in enumerate(x_in)])
        x_in = _convert(x_in, self._get_x_dtype())

        if isinstance(y_in, [tuple, list]):
            x_in = tuple([_convert(t, self._get_y_dtype(i)) for i, t in enumerate(y_in)])
        x_in = _convert(x_in, self._get_y_dtype())

        return x_in, y_in

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
            if self.classification:
                return np.argmax(y_, 1)
            return y_

        if not training:
            return _convert(y_pred), _convert(y_true)
        else:
            return y_pred, y_true
