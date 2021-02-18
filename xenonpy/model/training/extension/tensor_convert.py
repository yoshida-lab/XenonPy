#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Union, Tuple, Any, Sequence
from scipy.special import softmax

import numpy as np
import pandas as pd
import torch

from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension

__all__ = ['TensorConverter']

T_Data = Union[pd.DataFrame, pd.Series, np.ndarray, torch.Tensor]


class TensorConverter(BaseExtension):

    def __init__(self,
                 *,
                 x_dtype: Union[torch.dtype, Sequence[torch.dtype]] = None,
                 y_dtype: Union[torch.dtype, Sequence[torch.dtype]] = None,
                 empty_cache: bool = False,
                 auto_reshape: bool = True,
                 argmax: bool = False,
                 probability: bool = False):
        """
        Covert tensor like data into :class:`torch.Tensor` automatically.
        
        Parameters
        ----------
        x_dtype
            The :class:`torch.dtype`s of **X** data.
            If ``None``, will convert all data into ``torch.get_default_dtype()`` type.
            Can be a tuple of ``torch.dtype`` when your **X** is a tuple.
        y_dtype
            The :class:`torch.dtype`s of **y** data.
            If ``None``, will convert all data into ``torch.get_default_dtype()`` type.
            Can be a tuple of ``torch.dtype`` when your **y** is a tuple.
        empty_cache
            See Also: https://pytorch.org/docs/stable/cuda.html#torch.cuda.empty_cache
        auto_reshape
            Reshape tensor to (-1, 1) if tensor shape is (n,). Default ``True``.
        argmax
            Apply ``np.argmax(out, 1)`` on the output. This should only be used with classification model.
            Default ``False``. If ``True``, will ignore ``probability`` parameter.
        probability
            Apply ``scipy.special.softmax`` on the output. This should only be used with classification model.
            Default ``False``.
        """

        self.argmax = argmax
        self.empty_cache = empty_cache
        self.probability = probability
        self.auto_reshape = auto_reshape
        if x_dtype is None:
            self._x_dtype = torch.get_default_dtype()
        else:
            self._x_dtype = x_dtype

        if y_dtype is None:
            self._y_dtype = torch.get_default_dtype()
        else:
            self._y_dtype = y_dtype

    @property
    def argmax(self):
        return self._argmax

    @argmax.setter
    def argmax(self, value):
        self._argmax = value

    @property
    def empty_cache(self):
        return self._empty_cache

    @empty_cache.setter
    def empty_cache(self, value):
        self._empty_cache = value

    @property
    def auto_reshape(self):
        return self._auto_shape

    @auto_reshape.setter
    def auto_reshape(self, value):
        self._auto_shape = False if self.argmax or self.probability else value

    @property
    def probability(self):
        return self._probability

    @probability.setter
    def probability(self, value):
        self._probability = value

    def _get_x_dtype(self, i=0):
        if isinstance(self._x_dtype, tuple):
            return self._x_dtype[i]
        return self._x_dtype

    def _get_y_dtype(self, i=0):
        if isinstance(self._y_dtype, tuple):
            return self._y_dtype[i]
        return self._y_dtype

    def input_proc(self, x_in: Union[Sequence[Union[torch.Tensor, pd.DataFrame, pd.Series,
                                                    np.ndarray, Any]], torch.Tensor, pd.DataFrame,
                                     pd.Series, np.ndarray, Any],
                   y_in: Union[Sequence[Union[torch.Tensor, pd.DataFrame, pd.Series, np.ndarray,
                                              Any]], torch.Tensor, pd.DataFrame, pd.Series,
                               np.ndarray, Any],
                   trainer: Trainer) -> Tuple[torch.Tensor, torch.Tensor]:
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
            if len(t.size()) == 1 and self.auto_reshape:
                t = t.unsqueeze(1)
            return t.to(trainer.device, non_blocking=trainer.non_blocking)

        if isinstance(x_in, Sequence):
            x_in = tuple([_convert(t, self._get_x_dtype(i)) for i, t in enumerate(x_in)])
        else:
            x_in = _convert(x_in, self._get_x_dtype())

        if isinstance(y_in, Sequence):
            y_in = tuple([_convert(t, self._get_y_dtype(i)) for i, t in enumerate(y_in)])
        else:
            y_in = _convert(y_in, self._get_y_dtype())

        return x_in, y_in

    def step_forward(self):
        if self._empty_cache:
            torch.cuda.empty_cache()

    def output_proc(
        self,
        y_pred: Union[Sequence[Union[torch.Tensor, np.ndarray, Any]], torch.Tensor, Any],
        y_true: Union[Sequence[Union[torch.Tensor, np.ndarray, Any]], torch.Tensor, Any, None],
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

        def _convert(y_, argmax_=False, proba_=False):
            if y_ is None:
                return y_
            else:
                y_ = y_.detach().cpu().numpy()
            if argmax_:
                return np.argmax(y_, 1)
            if proba_:
                return softmax(y_, axis=1)
            return y_

        if not training:
            if isinstance(y_pred, tuple):
                return tuple([_convert(t, self._argmax, self._probability) for t in y_pred]), _convert(y_true)
            else:
                return _convert(y_pred, self._argmax, self._probability), _convert(y_true)
        else:
            return y_pred, y_true
