#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from copy import deepcopy
from typing import Union, Tuple, Any, Dict

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from xenonpy.model.training.base import BaseRunner

__all__ = ['Predictor']


class Predictor(BaseRunner):
    def __init__(self,
                 model: Module,
                 *,
                 cuda: Union[bool, str, torch.device] = False,
                 check_points: Dict[int, Dict] = None,
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
        self._checkpoints: Dict[int, Dict] = check_points if check_points else {}
        self._model = model.to(self._device)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: Module):
        """"""
        self._model = model.to(self._device)

    def __call__(self,
                 x_in: Union[Any, Tuple[Any]] = None,
                 y_true: Union[Any, Tuple[Any]] = None,
                 *,
                 dataset: DataLoader = None,
                 **model_params):
        """

        Parameters
        ----------
        x_in: pandas.DataFrame, numpy.ndarray, torch.Tensor
            Input data for test.
        y_true : Union[Any, Tuple[Any]]
        dataset : DataLoader
        model_params: dict
            Model parameters for prediction.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """
        return self.predict(x_in=x_in, y_true=y_true, dataset=dataset, **model_params)

    def predict(self,
                x_in: Union[Any, Tuple[Any]] = None,
                y_true: Union[Any, Tuple[Any]] = None,
                *,
                dataset: DataLoader = None,
                check_point: Union[int, str] = None,
                **model_params):
        """
        Predict from x input.
        This is just a simple wrapper for :meth:`~model.nn.utils.Predictor.predict`.

        Parameters
        ----------
        y_true
        dataset
        check_point
        x_in: Union[Any, Tuple[Any]]
            Input data for prediction..
        model_params: dict
            Model parameters for prediction.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """

        def _predict(x_, y_=None):
            x_, y_ = self.input_proc(x_, y_, )
            if not isinstance(x_, tuple):
                x_ = (x_,)

            if check_point:
                cp = self._checkpoints[check_point]
                model = deepcopy(self._model.cpu()).to(self._device)
                model.load_state_dict(cp['check_point'])
                y_p_ = model(*x_, **model_params)
            else:
                y_p_ = self._model(*x_, **model_params)

            return self.output_proc(y_p_, y_, )

        def _vstack(ls):
            if isinstance(ls[0], np.ndarray):
                return np.vstack(ls)
            if isinstance(ls[0], torch.Tensor):
                return torch.cat(ls, dim=0)
            return ls

        self._model.eval()
        if dataset is not None:
            y_preds = []
            y_trues = []
            for x_in, y_true in dataset:
                y_pred, y_true = _predict(x_in, y_true)
                y_preds.append(y_pred)
                y_trues.append(y_true)
            return _vstack(y_preds), _vstack(y_trues)
        elif x_in is not None and dataset is None:
            y_preds, y_trues = _predict(x_in, y_true)
            if y_trues is None:
                return y_preds
            return y_preds, y_trues
        else:
            raise RuntimeError('parameters <x_in> and <dataset> are mutually exclusive')
