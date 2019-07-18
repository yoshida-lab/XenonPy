#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union, Tuple

import torch
from sklearn.base import BaseEstimator
from torch.nn import Module

from xenonpy.utils import TimedMetaClass


class Predictor(BaseEstimator, metaclass=TimedMetaClass):
    def __init__(self,
                 model: Module,
                 *,
                 cuda: Union[bool, str] = False,
                 verbose: bool = True,
                 ):
        self.verbose = verbose
        self._device = self._check_cuda(cuda)
        self._model = model

    def _to_device(self, *tensor: torch.Tensor):
        return tuple([t.to(self._device) for t in tensor])

    @staticmethod
    def _check_cuda(cuda: Union[bool, str]) -> torch.device:
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

    def predict(self, x_test: Union[torch.Tensor, Tuple[torch.Tensor]], *, to_cpu=True, only_prediction=True):
        """

        Parameters
        ----------
        x_test: DataFrame, ndarray
            Input data for test.
        to_cpu: bool
            Should or not to return the prediction as numpy object.

        Returns
        -------
        any
            Return ::meth:`post_predict` results.
        """
        # prepare data
        if not isinstance(x_test, tuple):
            x_test = (x_test,)
        x_test = self._to_device(*x_test)

        # prediction
        self._model.to(self._device)
        self._model.eval()

        y_pred = self._model(*x_test)
        model_params = None
        if isinstance(y_pred, tuple):
            model_params = y_pred[1]
            y_pred = y_pred[0].detach()

            for k, v in model_params.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach()
                    if to_cpu:
                        v = v.cpu().numpy()
                    model_params[k] = v

        if to_cpu:
            y_pred = y_pred.cpu().numpy()

        if model_params is not None:
            return y_pred, model_params
        return y_pred
