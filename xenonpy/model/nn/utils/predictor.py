#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union, Tuple, Any

import numpy as np
import torch
from sklearn.base import BaseEstimator
from torch.nn import Module

from xenonpy.model.nn.utils.data_tool import T_Data, to_tensor, check_cuda
from xenonpy.utils.useful_cls import TimedMetaClass

__all__ = ['Predictor', 'T_Prediction']

T_Prediction = Union[np.ndarray, Tuple[np.ndarray, Any]]


class Predictor(BaseEstimator, metaclass=TimedMetaClass):
    def __init__(self,
                 model: Module,
                 *,
                 cuda: Union[bool, str, torch.device] = False,
                 verbose: bool = True,
                 ):
        self.verbose = verbose
        self._device = check_cuda(cuda)
        self._model = model

        self._model.to(self._device)
        self._model.eval()

    def _to_device(self, *tensor: torch.Tensor):
        return tuple([t.to(self._device) for t in tensor])

    def __call__(self, x: Union[T_Data, Tuple[T_Data]], *, predict_only: bool = True) -> T_Prediction:
        """
        Wrapper for :meth:`~Prediction.predict`.

        Parameters
        ----------
        x: DataFrame, ndarray
            Input data for test.
        predict_only: bool
            If ``False``, will returns all whatever the model returns.
            This can be useful for RNN models because these model also
            return `hidden variables` for recurrent training.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """
        return self.predict(x=x, predict_only=predict_only)

    def predict(self, x: Union[T_Data, Tuple[T_Data]], *, predict_only: bool = True) -> T_Prediction:
        """
        Predict values using given model.

        Parameters
        ----------
        x: DataFrame, ndarray
            Input data for test.
        predict_only: bool
            If ``False``, will returns all whatever the model returns.
            This can be useful for RNN models because these model also
            return `hidden variables` for recurrent training.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """
        # prepare data
        if not isinstance(x, tuple):
            x = (to_tensor(x),)
        else:
            x = tuple([to_tensor(x_) for x_ in x])

        # move tensor device
        x = self._to_device(*x)
        model_params = None

        y_pred = self._model(*x)
        if isinstance(y_pred, tuple):
            model_params = y_pred[1]
            y_pred = y_pred[0].detach().cpu().numpy()

            if not predict_only:
                for k, v in model_params.items():
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                        model_params[k] = v
        else:
            y_pred = y_pred.detach().cpu().numpy()

        if model_params is not None and not predict_only:
            return y_pred, model_params
        return y_pred
