#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict, namedtuple
from copy import deepcopy
from typing import Union, Tuple, List, Any, Dict, Callable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from xenonpy.model.training import ClipValue, ClipNorm
from xenonpy.model.training.base import BaseOptimizer, BaseLRScheduler, BaseRunner
from xenonpy.utils import camel_to_snake

__all__ = ['Trainer']


class Trainer(BaseRunner):
    checkpoint_tuple = namedtuple('checkpoint', 'id iterations model_state optimizer_state')
    results_tuple = namedtuple('results', 'total_epochs device training_info checkpoints model')

    def __init__(self,
                 *,
                 loss_func: Module,
                 optimizer: BaseOptimizer,
                 model: Module = None,
                 lr_scheduler: BaseLRScheduler = None,
                 clip_grad: Union[ClipNorm, ClipValue] = None,
                 epochs: int = 200,
                 cuda: Union[bool, str, torch.device] = False,
                 non_blocking: bool = False,
                 ):
        """
        NN model trainer.

        Parameters
        ----------
        loss_func: Tensor
            Loss function.
        optimizer: BaseOptimizer
            Optimizer for model parameters tuning.
        model: Module
            Pytorch NN model.
        lr_scheduler: BaseLRScheduler
            Learning rate scheduler.
        clip_grad : Union[ClipNorm, ClipValue]
            Clip grad before each optimize.
        epochs: int
            Number of iterations.
        cuda: Union[bool, str, torch.device]
            Set training device(s).
        """
        super().__init__(cuda=cuda)
        self._loss_func = loss_func
        self._loss_type = 'train_' + camel_to_snake(loss_func.__class__.__name__)
        self._clip_grad = clip_grad
        self.epochs = epochs
        self.non_blocking = non_blocking

        # set model
        self._model = None
        self._init_states = None
        self._set_model(model)

        # set optimizer
        self._optim = optimizer
        self._optimizer = None
        self._init_optim = None
        self._set_optimizer()

        # set lr_scheduler
        self._scheduler = lr_scheduler
        self._lr_scheduler = None
        self._set_lr_scheduler(lr_scheduler)

        # init private vars
        self._early_stopping: Tuple[bool, str] = (False, '')
        self._checkpoints: Dict[Union[int, str], Trainer.checkpoint_tuple] = OrderedDict()
        self._training_info: List[OrderedDict] = []
        self._total_its: int = 0  # of total iterations
        self._total_epochs: int = 0  # of total epochs

        self._x_val = None
        self._y_val = None
        self._validate_dataset = None

    def _set_model(self, model):
        if model is not None:
            self._model = model.to(self._device, non_blocking=self.non_blocking)
            self._init_states = deepcopy(model.state_dict())

    def _set_lr_scheduler(self, lr_scheduler):
        if lr_scheduler is not None and self._optimizer is not None:
            self._scheduler = lr_scheduler
            self._lr_scheduler: Union[_LRScheduler, None] = self._scheduler(self._optimizer)

    def _set_optimizer(self, optim=None):
        if optim is not None:
            self._optim = optim

        if self._model is not None:
            self._optimizer = self._optim(self._model.parameters())
            self._init_optim = deepcopy(self._optimizer.state_dict())

    @property
    def loss_type(self):
        return self._loss_type

    @property
    def total_epochs(self):
        return self._total_epochs

    @property
    def total_iterations(self):
        return self._total_its

    @property
    def x_val(self):
        return self._x_val

    @property
    def y_val(self):
        return self._y_val

    @property
    def validate_dataset(self):
        return self._validate_dataset

    @property
    def loss_func(self):
        return self._loss_func

    @property
    def training_info(self):
        if self._training_info:
            return pd.DataFrame(data=self._training_info)
        return None

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

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._set_optimizer(optimizer)

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, scheduler):
        self._set_lr_scheduler(scheduler)

    @property
    def clip_grad(self):
        return self._clip_grad

    @clip_grad.setter
    def clip_grad(self, fn):
        self._clip_grad = fn

    @property
    def checkpoints(self):
        return self._checkpoints

    def get_checkpoint(self, checkpoint: Union[int, str] = None):
        if checkpoint is None:
            return list(self._checkpoints.keys())
        if isinstance(checkpoint, int):
            id_ = f'cp:{checkpoint}'
            return self._checkpoints[id_]
        if isinstance(checkpoint, str):
            return self._checkpoints[checkpoint]
        raise TypeError(f'parameter <cp> must be str or int but got {checkpoint.__class__}')

    def set_checkpoint(self, id_: str = None):
        if id_ is None:
            id_ = f'cp:{self._total_its}'
        cp = self.checkpoint_tuple(
            id=id_,
            iterations=self._total_its,
            model_state=deepcopy(self._model.state_dict()),
            optimizer_state=deepcopy(self._optimizer.state_dict()),
        )

        self._checkpoints[id_] = cp
        self._on_checkpoint(checkpoint=cp, trainer=self, training=True)

    def early_stop(self, msg: str):
        self._early_stopping = (True, msg)

    def reset(self, *, to: Union[Module, int, str] = None):
        """
        Reset trainer.
        This will reset all trainer states and drop all training step information.

        Parameters
        ----------
        to: Union[bool, Module]
            Bind trainer to the given model or reset current model to it's initialization states.
        """
        self._training_info = []
        self._total_its = 0
        self._total_epochs = 0
        self._early_stopping = (False, '')

        if isinstance(to, Module):
            self._set_model(to)
            self._set_optimizer()
            self._set_lr_scheduler(self._scheduler)
            self._checkpoints = OrderedDict()
        elif isinstance(to, (int, str)):
            cp = self.get_checkpoint(to)
            self._model.load_state_dict(cp.model_state)
            self._optimizer.load_state_dict(cp.optimizer_state)
        elif to is None:
            self._model.load_state_dict(self._init_states)
            self._optimizer.load_state_dict(self._init_optim)
            self._checkpoints = OrderedDict()
        else:
            raise TypeError(f'parameter <to> must be torch.nnModule, int, or str but got {type(to)}')

        self._on_reset(trainer=self, training=True)

    def fit(self,
            x_train: Union[Any, Tuple[Any]] = None,
            y_train: Any = None,
            x_val: Union[Any, Tuple[Any]] = None,
            y_val: Any = None,
            *,
            training_dataset: DataLoader = None,
            validation_dataset: DataLoader = None,
            epochs: int = None,
            checkpoint: Union[bool, int, Callable[[int], Tuple[bool, str]]] = None,
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
        x_val : Union[Any, Tuple[Any]]
            Data for validation.
        y_val : Any
            Data for validation.
        validation_dataset: DataLoader
        epochs : int
            Epochs. If not ``None``, it will overwrite ``self.epochs`` temporarily.
        checkpoint: Union[bool, int, Callable[[int], bool]]
            If ``True``, will save model states at each step.
            If ``int``, will save model states every `checkpoint` steps.
            If ``Callable``, the function should take current ``total_epochs`` as input return ``bool``.
        model_params: dict
            Other model parameters.
        """
        if epochs is None:
            epochs = self.epochs

        # prob = self._total_epochs - 1
        with tqdm(total=epochs, desc='Training') as pbar:
            for _ in self(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, training_dataset=training_dataset,
                          validation_dataset=validation_dataset, epochs=epochs, checkpoint=checkpoint,
                          **model_params):
                # t = self._total_epochs - prob
                pbar.update()
                # prob = self._total_epochs

    def __call__(self,
                 x_train: Union[Any, Tuple[Any]] = None,
                 y_train: Any = None,
                 x_val: Union[Any, Tuple[Any]] = None,
                 y_val: Any = None,
                 *,
                 training_dataset: DataLoader = None,
                 validation_dataset: DataLoader = None,
                 epochs: int = None,
                 checkpoint: Union[bool, int, Callable[[int], bool]] = None,
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
        x_val : Union[Any, Tuple[Any]]
            Data for validation.
        y_val : Any
            Data for validation.
        validation_dataset : DataLoader
        epochs : int
            Epochs. If not ``None``, it will overwrite ``self.epochs`` temporarily.
        checkpoint: Union[bool, int, Callable[[int], bool]]
            If ``True``, will save model states at each step.
            If ``int``, will save model states every `checkpoint` steps.
            If ``Callable``, the function should take current ``total_epochs`` as input return ``bool``.
        model_params: dict
            Other model parameters.

        Yields
        ------
        namedtuple

        """
        if self._model is None:
            raise RuntimeError(
                'no model to train, use `trainer.model = <model>` or `trainer.reset(to=<model>)` to bind a model')

        if epochs is None:
            epochs = self.epochs

        self._model.to(self._device, non_blocking=self.non_blocking)

        if training_dataset is not None:
            if y_train is not None or x_train is not None:
                raise RuntimeError('parameter <training_dataset> is exclusive of <x_train> and <y_train>')
        else:
            if y_train is None or x_train is None:
                raise RuntimeError('missing parameter <x_train> or <y_train>')

        # training step
        def _step(x_, y_, i_b=0):
            def closure():
                self._optimizer.zero_grad()
                y_p_ = self._model(*x_, **model_params)
                y_p_, y_t_ = self.output_proc(y_p_, y_, trainer=self, training=True)
                loss_ = self._loss_func(y_p_, y_t_)
                loss_.backward()

                if self._clip_grad is not None:
                    self._clip_grad(self._model.parameters())

                return loss_

            train_loss = self._optimizer.step(closure).item()

            step_info = OrderedDict(
                total_iters=self._total_its,
                i_epoch=self._total_epochs,
                i_batch=i_b + 1, )
            step_info[self._loss_type] = train_loss
            self._total_its += 1
            self._step_forward(step_info=step_info, trainer=self, training=True)
            self._training_info.append(step_info)

            if self._lr_scheduler is not None and isinstance(self._lr_scheduler, ReduceLROnPlateau):
                self._lr_scheduler.step(train_loss, epoch=self._total_epochs)

            return step_info

        def _snapshot():
            if checkpoint is not None:
                if isinstance(checkpoint, bool) and checkpoint:
                    self.set_checkpoint()
                if isinstance(checkpoint, int):
                    if self._total_epochs % checkpoint == 0:
                        self.set_checkpoint()
                if callable(checkpoint):
                    flag, msg = checkpoint(self._total_epochs)
                    if flag:
                        self.set_checkpoint(msg)

        if validation_dataset is not None:
            if y_val is not None or x_val is not None:
                raise RuntimeError('parameter <validation_dataset> is exclusive of <x_val> and <y_val>')
            else:
                self._validate_dataset = validation_dataset
        else:
            if y_val is not None and x_val is not None:
                self._x_val, self._y_val = self.input_proc(x_val, y_val, trainer=self, training=False)

        # before processing
        self._before_proc(trainer=self, training=True)

        if training_dataset:
            for i_epoch in range(self._total_epochs, epochs + self._total_epochs):

                # decay learning rate
                if self._lr_scheduler is not None and not isinstance(self._lr_scheduler, ReduceLROnPlateau):
                    self._lr_scheduler.step(epoch=self._total_its)

                self._model.train()
                self._total_epochs += 1
                for i_batch, (x_train, y_train) in enumerate(training_dataset):
                    x_train, y_train = self.input_proc(x_train, y_train, trainer=self, training=True)
                    if not isinstance(x_train, tuple):
                        x_train = (x_train,)
                    yield _step(x_train, y_train, i_batch)
                    if self._early_stopping[0]:
                        tqdm.write(f'Early stopping is applied: {self._early_stopping[1]}')
                        self._after_proc(trainer=self, training=True)
                        self._model.eval()
                        return
                _snapshot()

        else:
            x_train, y_train = self.input_proc(x_train, y_train, trainer=self, training=True)
            if not isinstance(x_train, tuple):
                x_train = (x_train,)

            for i_epoch in range(self._total_epochs, epochs + self._total_epochs):

                if self._lr_scheduler is not None and not isinstance(self._lr_scheduler, ReduceLROnPlateau):
                    self._lr_scheduler.step(epoch=self._total_its)

                self._model.train()
                self._total_epochs += 1
                yield _step(x_train, y_train)
                if self._early_stopping[0]:
                    tqdm.write(f'Early stopping is applied: {self._early_stopping[1]}.')
                    self._after_proc(trainer=self, training=True)
                    self._model.eval()
                    return
                _snapshot()

        # after processing
        self._after_proc(trainer=self, training=True)
        self._model.eval()

    def predict(self,
                x_in: Union[Any, Tuple[Any]] = None,
                y_true: Union[Any, Tuple[Any]] = None,
                *,
                dataset: DataLoader = None,
                checkpoint: Union[int, str] = None,
                **model_params):
        """
        Predict from x input.
        This is just a simple wrapper for :meth:`~model.nn.utils.Predictor.predict`.

        Parameters
        ----------
        checkpoint: Union[int, str]
        x_in: Union[Any, Tuple[Any]]
            Input data for prediction.
        y_true: Union[Any, Tuple[Any]]
        dataset: DataLoader
        model_params: dict
            Model parameters for prediction.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """

        def _predict(x_, y_=None):
            x_, y_ = self.input_proc(x_, y_, trainer=self, training=False)
            if not isinstance(x_, tuple):
                x_ = (x_,)

            if checkpoint:
                cp = self.get_checkpoint(checkpoint)
                model = deepcopy(self._model).to(self.device)
                model.load_state_dict(cp.model_state)
                y_p_ = model(*x_, **model_params)
            else:
                y_p_ = self._model(*x_, **model_params)

            return self.output_proc(y_p_, y_, trainer=self, training=False)

        def _vstack(ls):
            if isinstance(ls[0], np.ndarray):
                return np.vstack(ls)
            if isinstance(ls[0], torch.Tensor):
                return torch.cat(ls, dim=0)
            return ls

        self._model.eval()
        self._model.to(self.device, non_blocking=self.non_blocking)
        if x_in is None and y_true is None and dataset is not None:
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

    def to_namedtuple(self):
        return self.results_tuple(
            total_epochs=self.total_epochs,
            device=self.device,
            training_info=self.training_info,
            checkpoints={k: deepcopy(v.model_state) for k, v in self._checkpoints.items()},
            model=deepcopy(self._model.cpu())
        )
