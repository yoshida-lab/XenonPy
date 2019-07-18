#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from typing import Union, Tuple, Callable

import pandas as pd
import torch
from sklearn.base import BaseEstimator
from torch.nn import Module
from torch.utils.data import DataLoader

from xenonpy.model.nn.utils.base import BaseExtension, check_cuda
from xenonpy.model.nn.wrap.base import BaseOptimizer, BaseLRScheduler
from xenonpy.utils import TimedMetaClass


class Trainer(BaseEstimator, metaclass=TimedMetaClass):

    def __init__(self,
                 model: Module,
                 loss_func: Module,
                 optimizer: BaseOptimizer,
                 *,
                 lr_scheduler: BaseLRScheduler = None,
                 model_modifier: Callable[[Module, ], None] = None,
                 epochs: int = 2000,
                 cuda: Union[bool, str] = False,
                 verbose: bool = True,
                 ):
        self.model_modifier = model_modifier
        self.epochs = epochs
        self.loss_func = loss_func
        self.optimizer = optimizer(self._model.parameters())
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

        self.verbose = verbose
        self._device = check_cuda(cuda)
        self._model = model

        self._extensions = []
        self._model_states = []
        self._step_info = []

    def _to_device(self, *tensor: torch.Tensor):
        return tuple([t.to(self._device) for t in tensor])

    @property
    def losses(self):
        return pd.DataFrame(data=self._step_info)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, v):
        self._device = check_cuda(v)

    @property
    def elapsed(self):
        return self._timer.elapsed

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        if isinstance(m, torch.nn.Module):
            self._model = m
        else:
            raise TypeError(
                'parameter `m` must be a instance of <torch.nn.modules> but got %s' % type(m))

    def fit(self,
            x_train: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
            y_train: torch.Tensor = None,
            *,
            training_dataset: DataLoader = None,
            yield_step: bool = False,
            save_training_state: bool = False,
            model_params: dict = None):
        """
        Train the Neural Network model

        Parameters
        ----------
        x_train: Union[torch.Tensor, Tuple[torch.Tensor]]
            Training data. Will be ignored will``training_dataset`` is given.
        y_train: torch.Tensor
            Test data. Will be ignored will``training_dataset`` is given.
        training_dataset: DataLoader
            Torch DataLoader. If given, will only use this as training dataset.
            When loop over this dataset, it should yield a tuple contains ``x_train`` and ``y_train`` in order.
        yield_step : False
            Yields intermediate information.
        model_params: dict
            Other model parameters.
        save_training_state: bool
            If ``True``, will save model status on each training step_info.

        Yields
        ------
        namedtuple

        """

        if model_params is None:
            model_params = {}

        for k, v in model_params.items():
            if isinstance(v, torch.Tensor):
                model_params[k] = v.to(self._device)

        self._model.to(self._device)
        self._model.train()

        if training_dataset is not None:
            if y_train is not None or x_train is not None:
                raise RuntimeError('parameter <data_loader> is exclusive of <x_train> and <y_train>')
        else:
            if y_train is None or x_train is None:
                raise RuntimeError('missing parameter <x_train> or <y_train>')

        for i_epoch in range(self.epochs):
            if training_dataset:
                for i_batch, (x, y) in enumerate(training_dataset):

                    # convert to tuple for convenient
                    if not isinstance(x, tuple):
                        x = (x,)

                    # move tensor device
                    x = self._to_device(*x)
                    y = y.to(self._device)

                    # feed model
                    i_closure = [0]  # set iteration counter for closure

                    def closure():
                        i_closure[0] = i_closure[0] + 1
                        self.optimizer.zero_grad()
                        y_pred_ = self._model(*x, **model_params)
                        if isinstance(y_pred_, tuple):
                            model_params.update(y_pred_[1])
                            y_pred_ = y_pred_[0]
                        loss_ = self.loss_func(y_pred_, y)
                        loss_.backward()

                        if self.model_modifier is not None:
                            self.model_modifier(self.model)

                        return loss_

                    train_loss = self.optimizer.step(closure).item() / y.size(0)
                    i_closure = [0]  # reset counter

                    step_info = OrderedDict(i_epoch=i_epoch, i_batch=i_batch, train_loss=train_loss)
                    step_info = self._exec_ext(step_info)

                    self._step_info.append(step_info)

                    if save_training_state:
                        self._model_states.append(self._model.state_dict())

                    if yield_step:
                        yield step_info

            else:
                if not isinstance(x_train, tuple):
                    x_train = (x_train,)

                # move tensor device
                x = self._to_device(*x_train)
                y = y_train.to(self._device)

                # feed model
                i_closure = [0]

                def closure():
                    i_closure[0] = i_closure[0] + 1
                    self.optimizer.zero_grad()
                    y_pred_ = self._model(*x, **model_params)
                    if isinstance(y_pred_, tuple):
                        model_params.update(y_pred_[1])
                        y_pred_ = y_pred_[0]
                    loss_ = self.loss_func(y_pred_, y)
                    loss_.backward()

                    if self.model_modifier is not None:
                        self.model_modifier(self.model, loss=loss_)

                    return loss_

                train_loss = self.optimizer.step(closure).item() / y.size(0)
                i_closure = [0]

                step_info = OrderedDict(i_epoch=i_epoch, train_loss=train_loss)
                step_info = self._exec_ext(step_info)

                self._step_info.append(step_info)

                if save_training_state:
                    self._model_states.append(self._model.state_dict())

                if yield_step:
                    yield step_info

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

    def _exec_ext(self, step_info):
        for ext in self._extensions:
            step_info = ext(step_info, self)
        return step_info

    def extend(self, ext: Callable[[OrderedDict, BaseExtension], OrderedDict]):
        self._extensions.append(ext)

    def as_dict(self):
        return OrderedDict(
            epoches=self.epochs,
            losses=self._step_info,
            states=self._model_states,
            model=type(self._model)().load_state_dict(self._model.state_dict())
        )
