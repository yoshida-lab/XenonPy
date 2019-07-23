#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, Callable, List, Any

import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from xenonpy.model.training.base import BaseOptimizer, BaseLRScheduler, BaseRunner
from xenonpy.model.utils import Predictor

__all__ = ['Trainer']


class Trainer(BaseRunner):

    def __init__(self,
                 model: Module,
                 loss_func: Module,
                 optimizer: BaseOptimizer,
                 *,
                 epochs: int = 2000,
                 cuda: Union[bool, str, torch.device] = False,
                 lr_scheduler: BaseLRScheduler = None,
                 model_modifier: Callable[[Module, float], None] = None,
                 verbose: bool = True,
                 ):
        """
        NN model trainer.

        Parameters
        ----------
        model: Module
            Pytorch NN model.
        loss_func: Tensor
            Loss function.
        optimizer: BaseOptimizer
            Optimizer for model parameters tuning.
        epochs: int
            Number of iterations.
        cuda: Union[bool, str, torch.device]
            Set training device(s).
        lr_scheduler: BaseLRScheduler
            Learning rate scheduler.
        model_modifier : Callable[[Module, float], None]
            Modify model parameters before each optimize.
        verbose: bool
            Wither to use verbose output.
        """
        super().__init__(cuda=cuda)
        self._model = model.to(self._device)
        self.loss_func = loss_func
        self.optimizer = optimizer(self._model.parameters())

        # optional
        self.epochs = epochs
        self._scheduler = lr_scheduler
        if lr_scheduler is not None:
            self.lr_scheduler: Union[_LRScheduler, None] = lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler: Union[_LRScheduler, None] = None
        self.verbose = verbose
        self.model_modifier = model_modifier

        # init private vars
        self._optim = optimizer
        self._init_states = model.state_dict()
        self._model_states = []
        self._step_info: List[OrderedDict] = []
        self._total_iters: int = 0

        # others
        self._predictor = Predictor(self._model, cuda=self._device)

    @property
    def losses(self):
        if self._step_info:
            return pd.DataFrame(data=self._step_info)
        return None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, v):
        self._device = self.check_cuda(v)

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

    def reset(self, *, model: Union[bool, Module] = False):
        """
        Reset trainer.
        This will reset all trainer states and drop all training step information.

        Parameters
        ----------
        model: Union[bool, Module]
            Bind trainer to the given model or reset current model to it's initialization states.
        """
        self._step_info = []
        self._model_states = []
        self._total_iters = 0

        if model is not False:
            if isinstance(model, Module):
                self._model = model
                self._init_states = model.state_dict()
            else:
                self._model.load_state_dict(self._init_states)

            self.optimizer = self._optim(self._model.parameters())
            if self._scheduler is not None:
                self.lr_scheduler: Union[_LRScheduler, None] = self._scheduler(self.optimizer)
            else:
                self.lr_scheduler: Union[_LRScheduler, None] = None

    def fit(self,
            x_train: Union[Any, Tuple[Any]] = None,
            y_train: Any = None,
            *,
            training_dataset: DataLoader = None,
            save_training_state: bool = False,
            **model_params):
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
        save_training_state: bool
            If ``True``, will save model status on each training step_info.
        model_params: dict
            Other model parameters.
        """
        for _ in self(x_train=x_train, y_train=y_train, training_dataset=training_dataset,
                      save_training_state=save_training_state, **model_params):
            continue

    def __call__(self,
                 x_train: Union[Any, Tuple[Any]] = None,
                 y_train: Any = None,
                 *,
                 epochs: int = None,
                 training_dataset: DataLoader = None,
                 save_training_state: bool = False,
                 **model_params):
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
        epochs : int
            Epochs. If not ``None``, it will overwrite ``self.epochs`` temporarily.
        save_training_state: bool
            If ``True``, will save model status on each training step_info.
        model_params: dict
            Other model parameters.

        Yields
        ------
        namedtuple

        """

        if epochs is None:
            epochs = self.epochs

        self._model.to(self._device)

        if training_dataset is not None:
            if y_train is not None or x_train is not None:
                raise RuntimeError('parameter <data_loader> is exclusive of <x_train> and <y_train>')
        else:
            if y_train is None or x_train is None:
                raise RuntimeError('missing parameter <x_train> or <y_train>')

        self._before_proc()

        if training_dataset:
            train_loss = 1e6
            for i_epoch in tqdm(range(self._total_iters, epochs + self._total_iters)):

                # decay learning rate
                if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(epoch=self._total_iters)

                self._model.train()

                for i_batch, (x, y) in enumerate(training_dataset):
                    x, y = self.input_proc(x, y)
                    if not isinstance(x, tuple):
                        x = (x,)

                    # feed model
                    def closure():
                        self.optimizer.zero_grad()
                        y_pred_ = self._model(*x, **model_params)
                        y_pred_ = self.output_proc(y_pred_)
                        loss_ = self.loss_func(y_pred_, y)
                        loss_.backward()

                        if self.model_modifier is not None:
                            self.model_modifier(self._model, loss=loss_)

                        return loss_

                    train_loss = self.optimizer.step_forward(closure).item() / y.size(0)

                    step_info = OrderedDict(i_epoch=i_epoch + 1, i_batch=i_batch + 1, train_loss=train_loss)
                    self._step_forward(step_info)
                    self._step_info.append(step_info)
                    self._total_iters = i_epoch + 1

                    if save_training_state:
                        self._model_states.append(self._model.state_dict())

                    yield step_info

                if self.lr_scheduler is not None and isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(train_loss, epoch=self._total_iters)

        else:
            x, y = self.input_proc(x_train, y_train)
            if not isinstance(x, tuple):
                x = (x,)

            for i_epoch in tqdm(range(self._total_iters, epochs + self._total_iters)):

                if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(epoch=self._total_iters)
                self._model.train()

                # feed model
                def closure():
                    self.optimizer.zero_grad()
                    y_pred_ = self._model(*x, **model_params)
                    y_pred_ = self.output_proc(y_pred_)
                    loss_ = self.loss_func(y_pred_, y)
                    loss_.backward()

                    if self.model_modifier is not None:
                        self.model_modifier(self._model, loss=loss_)

                    return loss_

                train_loss = self.optimizer.step_forward(closure).item() / y.size(0)

                step_info = OrderedDict(i_epoch=i_epoch + 1, train_loss=train_loss)
                self._step_forward(step_info)
                self._step_info.append(step_info)
                self._total_iters = i_epoch + 1

                if save_training_state:
                    self._model_states.append(self._model.state_dict())

                yield step_info

                if self.lr_scheduler is not None and isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(train_loss, epoch=self._total_iters)

        self._after_proc()
        self._model.cpu().eval()

    def predict(self, x: Union[Any, Tuple[Any]], **model_params):
        """
        Predict from x input.
        This is just a simple wrapper for :meth:`~model.nn.utils.Predictor.predict`.

        Parameters
        ----------
        x: Union[Any, Tuple[Any]]
            Input data for prediction..
        model_params: dict
            Model parameters for prediction.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """
        x = self.input_proc(x, train=False)
        y_pred = self._predictor(x, **model_params)
        return self.output_proc(y_pred, train=False)

    def as_dict(self):
        return dict(
            total_iteration=self._total_iters,
            losses=self._step_info,
            states=self._model_states,
            model=deepcopy(self._model.cpu())
        )
