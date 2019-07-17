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

from .wrap.base import BaseOptimizer, BaseLRScheduler
from ...utils import TimedMetaClass


class BaseTrainer(BaseEstimator, metaclass=TimedMetaClass):

    def __init__(self,
                 model: torch.nn.Module,
                 loss: Module,
                 optimizer: BaseOptimizer,
                 *,
                 lr_scheduler: BaseLRScheduler = None,
                 model_modifier: Callable[[Module], None] = None,
                 epochs: int = 2000,
                 cuda: Union[bool, str] = False,
                 verbose: bool = True,
                 ):
        self.model_modifier = model_modifier
        self.epochs = epochs
        self.loss_func = loss
        self.optimizer = optimizer(self._model.parameters())
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None

        self.verbose = verbose
        self._device = self._check_cuda(cuda)
        self._model = model

        self._states = []
        self._losses = []

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

    @property
    def losses(self):
        return pd.DataFrame(data=self._losses)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, v):
        self._device = self._check_cuda(v)

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
            x_test: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
            y_test: torch.Tensor = None,
            yield_step: bool = False,
            save_training_state: bool = False,
            model_params: dict = None):
        """
        Fit Neural Network model

        Parameters
        ----------
        x_train: Union[torch.Tensor, Tuple[torch.Tensor]]
            Training data. Will be ignored will``data_loader`` is given.
        y_train: torch.Tensor
            Test data. Will be ignored will``data_loader`` is given.
        training_dataset: .DataLoader
            Torch DataLoader. If given, will only use this as training dataset.
        y_test: Union[torch.Tensor, Tuple[torch.Tensor]]
        x_test: torch.Tensor
        yield_step : False
            Yields intermediate information.
        model_params: dict
            Other model parameters.
        save_training_state: bool
            If ``True``, will save model status on each training step.

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

                    step = OrderedDict(i_epoch=i_epoch, i_batch=i_batch, train_loss=train_loss)

                    if x_test is not None and y_test is not None:
                        y_pred, y = self.predict(x_test), y_test.to(self._device)
                        step['test_loss'] = self.loss_func(y_pred, y).item()
                        self._model.train()

                    self._losses.append(step)

                    if save_training_state:
                        self._states.append(self._model.state_dict())

                    if yield_step:
                        yield step

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
                        self.model_modifier(self.model)

                    return loss_

                train_loss = self.optimizer.step(closure).item() / y.size(0)
                i_closure = [0]

                step = OrderedDict(i_epoch=i_epoch, i_batch=0, train_loss=train_loss)

                if x_test is not None and y_test is not None:
                    y_pred, y = self.predict(x_test), y_test.to(self._device)
                    step['test_loss'] = self.loss_func(y_pred, y).item()
                    self._model.train()

                self._losses.append(step)

                if save_training_state:
                    self._states.append(self._model.state_dict())

                if yield_step:
                    yield step

    def predict(self, x_test: Union[torch.Tensor, Tuple[torch.Tensor]], *, to_cpu=True):
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

    def as_dict(self):
        return OrderedDict(
            epoches=self.epochs,
            losses=self._losses,
            states=self._states,
            model=type(self._model)().load_state_dict(self._model.state_dict())
        )
