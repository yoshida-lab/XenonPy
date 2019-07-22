#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union

import torch
from torch.nn import Module

from xenonpy.model.training.base import BaseRunner

__all__ = ['Predictor']


class Predictor(BaseRunner):
    def __init__(self,
                 model: Module,
                 *,
                 cuda: Union[bool, str, torch.device] = False,
                 verbose: bool = True,
                 ):
        """

        Parameters
        ----------
        model
        cuda
        verbose
        """
        super().__init__(cuda=cuda)
        self.verbose = verbose
        self._model = model.to(self._device)

    def __call__(self, x, **model_params):
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

    def predict(self, x, **model_params):
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
        self._model.eval()

        # prepare data
        x = self.input_proc(x, train=False)
        if not isinstance(x, tuple):
            x = (x,)

        # move tensor device
        y_pred = self._model(*x, **model_params)

        return self.output_proc(y_pred, train=False)
