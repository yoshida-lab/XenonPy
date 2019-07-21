#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union, Tuple, Any

import numpy as np
import torch
from torch.nn import Module

from xenonpy.model.nn.training.base import BaseRunner
from xenonpy.model.nn.utils.data_tool import T_Data, check_cuda

__all__ = ['Predictor', 'T_Prediction']

T_Prediction = Union[np.ndarray, Tuple[np.ndarray, Any]]


class Predictor(BaseRunner):
    def __init__(self,
                 model: Module,
                 *,
                 cuda: Union[bool, str, torch.device] = False,
                 verbose: bool = True,
                 ):
        super().__init__()
        self.verbose = verbose
        self._device = check_cuda(cuda)
        self._model = model

        self._model.to(self._device)
        self._model.eval()

    def pre_process(self, x_pred):
        for ext, _ in self._extensions:
            x_pred = ext.pre_process(x=x_pred)
        return x_pred

    def post_process(self, y_pred):
        for ext, _ in self._extensions:
            y_pred = ext.post_process(y_pred)
        return y_pred

    def _to_device(self, *tensor: torch.Tensor):
        return tuple([t.to(self._device) for t in tensor])

    def __call__(self, x: Union[T_Data, Tuple[T_Data]], **model_params) -> T_Prediction:
        """
        Wrapper for :meth:`~Prediction.predict`.

        Parameters
        ----------
        x: DataFrame, ndarray
            Input data for test.
        model_params: dict
            Model parameters for prediction.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """
        return self.predict(x=x, **model_params)

    def predict(self, x: Union[T_Data, Tuple[T_Data]], **model_params) -> T_Prediction:
        """
        Predict values using given model.

        Parameters
        ----------
        x: DataFrame, ndarray
            Input data for test.
        model_params: dict
            Model parameters for prediction.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """
        # prepare data
        x = self.pre_process(x)

        if not isinstance(x, tuple):
            x = (x,)

        # move tensor device
        x = self._to_device(*x)

        y_pred = self._model(*x, **model_params)
        return self.post_process(y_pred)
