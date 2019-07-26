#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, List, Any, Dict, Callable

import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from xenonpy.model.training import ClipValue, ClipNorm
from xenonpy.model.training import Predictor
from xenonpy.model.training.base import BaseOptimizer, BaseLRScheduler, BaseRunner

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
                 clip_grad: Union[ClipNorm, ClipValue] = None,
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
        clip_grad : Union[ClipNorm, ClipValue]
            Clip grad before each optimize.
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
        self.clip_grad = clip_grad

        # init private vars
        self._optim = optimizer
        self._init_states = deepcopy(model.state_dict())
        self._init_optim = deepcopy(self.optimizer.state_dict())
        self._check_points: Dict[int, Dict] = {}
        self._step_info: List[OrderedDict] = []
        self._total_its: int = 1  # of total iterations
        self._total_epochs: int = 1  # of total epochs

        # others
        self._predictor = Predictor(self._model, cuda=self._device)

    @property
    def step_info(self):
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
    def model(self):
        return self._model

    @model.setter
    def model(self, m):
        if isinstance(m, torch.nn.Module):
            self.reset(to=m)
        else:
            raise TypeError(
                'parameter `m` must be a instance of <torch.nn.modules> but got %s' % type(m))

    def reset(self, *, to: Union[Module, int, str] = None):
        """
        Reset trainer.
        This will reset all trainer states and drop all training step information.

        Parameters
        ----------
        to: Union[bool, Module]
            Bind trainer to the given model or reset current model to it's initialization states.
        """
        self._step_info = []
        self._check_points = {}
        self._total_its = 1
        self._total_epochs = 1

        if isinstance(to, Module):
            self._model = to
            self._init_states = deepcopy(to.state_dict())
            self._predictor = Predictor(self._model, cuda=self._device)
            self.optimizer = self._optim(self._model.parameters())
            if self._scheduler is not None:
                self.lr_scheduler: Union[_LRScheduler, None] = self._scheduler(self.optimizer)
            else:
                self.lr_scheduler: Union[_LRScheduler, None] = None
        elif isinstance(to, (int, str)):
            cp = self._check_points[to]
            self._model.load_state_dict(cp['check_point'])
            self.optimizer.load_state_dict(cp['optimizer'])
            self._total_epochs = cp['total_epochs']
            self._total_its = to
        else:
            self._model.load_state_dict(self._init_states)
            self.optimizer.load_state_dict(self._init_optim)

        self._reset_proc()

    def snapshot(self, name=None, **kwargs):
        cp = dict(
            total_epochs=self._total_epochs,
            total_iteration=self._total_its,
            check_point=deepcopy(self._model.state_dict()),
            optimizer=deepcopy(self.optimizer.state_dict()),
            **kwargs
        )
        if name is None:
            self._check_points[self._total_its] = cp
        else:
            self._check_points[name] = cp

    def fit(self,
            x_train: Union[Any, Tuple[Any]] = None,
            y_train: Any = None,
            *,
            epochs: int = None,
            training_dataset: DataLoader = None,
            check_point: Union[bool, int, Callable[[int], bool]] = None,
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
        check_point: Union[bool, int, Callable[[int], bool]]
            If ``True``, will save model states at each step.
            If ``int``, will save model states every `check_point` steps.
            If ``Callable``, the function should take current ``total_epochs`` as input return ``bool``.
        model_params: dict
            Other model parameters.
        """
        if epochs is None:
            epochs = self.epochs

        prob = self._total_epochs - 1
        with tqdm(total=epochs, desc='Training') as pbar:
            for _ in self(x_train=x_train, y_train=y_train, training_dataset=training_dataset, epochs=epochs,
                          check_point=check_point, **model_params):
                t = self._total_epochs - prob
                pbar.update(t)
                prob = self._total_epochs

    def __call__(self,
                 x_train: Union[Any, Tuple[Any]] = None,
                 y_train: Any = None,
                 *,
                 epochs: int = None,
                 training_dataset: DataLoader = None,
                 check_point: Union[bool, int, Callable[[int], bool]] = None,
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
        check_point: Union[bool, int, Callable[[int], bool]]
            If ``True``, will save model states at each step.
            If ``int``, will save model states every `check_point` steps.
            If ``Callable``, the function should take current ``total_epochs`` as input return ``bool``.
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

        # training step
        def _step(x, y, i_b=0):
            def closure():
                self.optimizer.zero_grad()
                y_pred_ = self._model(*x, **model_params)
                y_pred_ = self.output_proc(y_pred_)
                loss_ = self.loss_func(y_pred_, y)
                loss_.backward()

                if self.clip_grad is not None:
                    self.clip_grad(self._model.parameters())

                return loss_

            train_loss = self.optimizer.step(closure).item()

            step_info = OrderedDict(
                total_iters=self._total_its,
                i_epoch=self._total_epochs,
                i_batch=i_b + 1,
                train_loss=train_loss)

            self._step_forward(step_info)
            self._step_info.append(step_info)

            if self.lr_scheduler is not None and isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(train_loss, epoch=self._total_epochs)

            self._total_its += 1
            return step_info

        def _snapshot():
            if check_point is not None:
                if isinstance(check_point, bool) and check_point:
                    self.snapshot()
                if isinstance(check_point, int):
                    if self._total_epochs % check_point == 0:
                        self.snapshot()
                if callable(check_point):
                    if check_point(self._total_epochs):
                        self.snapshot()

        # before processing
        self._before_proc()

        if training_dataset:
            for i_epoch in range(self._total_epochs, epochs + self._total_epochs):

                # decay learning rate
                if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(epoch=self._total_its)

                self._model.train()
                for i_batch, (x_train, y_train) in enumerate(training_dataset):
                    x_train, y_train = self.input_proc(x_train, y_train)
                    if not isinstance(x_train, tuple):
                        x_train = (x_train,)
                    yield _step(x_train, y_train, i_batch)
                _snapshot()
                self._total_epochs += 1

        else:
            x_train, y_train = self.input_proc(x_train, y_train)
            if not isinstance(x_train, tuple):
                x_train = (x_train,)

            for i_epoch in range(self._total_epochs, epochs + self._total_epochs):

                if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, ReduceLROnPlateau):
                    self.lr_scheduler.step(epoch=self._total_its)

                self._model.train()
                yield _step(x_train, y_train)
                _snapshot()
                self._total_epochs += 1

        # after processing
        self._after_proc()
        self._model.cpu().eval()

    def predict(self, x: Union[Any, Tuple[Any]], *, check_point: Union[int, str] = None, **model_params):
        """
        Predict from x input.
        This is just a simple wrapper for :meth:`~model.nn.utils.Predictor.predict`.

        Parameters
        ----------
        check_point
        x: Union[Any, Tuple[Any]]
            Input data for prediction..
        model_params: dict
            Model parameters for prediction.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """
        x = self.input_proc(x, training=False)
        if check_point:
            cp = self._check_points[check_point]
            model = deepcopy(self._model.cpu())
            model.load_state_dict(cp['check_point'])
            self._predictor.model = model
            y_pred = self._predictor(x, **model_params)
            self._predictor.model = self._model
        else:
            y_pred = self._predictor(x, **model_params)
        return self.output_proc(y_pred, training=False)

    def as_dict(self, *, check_point: Union[int, str] = None):
        if check_point:
            cp = self._check_points[check_point]
            model = deepcopy(self._model.cpu())
            model.load_state_dict(cp['check_point'])
            return dict(
                total_iteration=cp['total_iteration'],
                total_epochs=cp['total_epochs'],
                step_info=self._step_info,
                model=model
            )

        return dict(
            total_iteration=self._total_its,
            total_epochs=self._total_epochs,
            step_info=self._step_info,
            model=deepcopy(self._model.cpu())
        )
