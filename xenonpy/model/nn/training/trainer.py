#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from copy import deepcopy
from typing import Union, Tuple, Callable, List

import pandas as pd
import torch
from sklearn.base import BaseEstimator
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from xenonpy.model.nn.training.base import BaseOptimizer, BaseLRScheduler, BaseExtension
from xenonpy.model.nn.utils import check_cuda, to_tensor, T_Data, Predictor, T_Prediction
from xenonpy.utils import TimedMetaClass

__all__ = ['Trainer']


class Trainer(BaseEstimator, metaclass=TimedMetaClass):

    def __init__(self,
                 model: Module,
                 loss_func: Module,
                 optimizer: BaseOptimizer,
                 *,
                 epochs: int = 2000,
                 cuda: Union[bool, str] = False,
                 lr_scheduler: BaseLRScheduler = None,
                 model_modifier: Callable[[Module, ], None] = None,
                 verbose: bool = True,
                 ):
        self._model = model
        self.loss_func = loss_func
        self.optimizer = optimizer(self._model.parameters())

        # optional
        self.epochs = epochs
        self._device = check_cuda(cuda)
        if lr_scheduler is not None:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        else:
            self.lr_scheduler = None
        self.model_modifier = model_modifier
        self.verbose = verbose

        # init container
        self._extensions: List[BaseExtension] = []
        self._model_states = []
        self._step_info = []
        self._total_iters = 0

        # others
        self._predictor = Predictor(self._model, cuda=self._device)

    def _to_device(self, *tensor: torch.Tensor):
        return tuple([t.to(self._device) for t in tensor])

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

    def reset(self):
        self._step_info = []
        self._model_states = []
        self._total_iters = 0

    def fit(self,
            x_train: Union[T_Data, Tuple[T_Data]] = None,
            y_train: T_Data = None,
            *,
            training_dataset: DataLoader = None,
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
        model_params: dict
            Other model parameters.
        save_training_state: bool
            If ``True``, will save model status on each training step_info.
        """
        for _ in self(x_train=x_train, y_train=y_train, training_dataset=training_dataset,
                      save_training_state=save_training_state, model_params=model_params):
            continue

    def __call__(self,
                 x_train: Union[T_Data, Tuple[T_Data]] = None,
                 y_train: T_Data = None,
                 *,
                 epochs: int = None,
                 training_dataset: DataLoader = None,
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
        epochs : int
            Epochs. If not ``None``, it will overwrite ``self.epochs`` temporarily.
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

        if epochs is None:
            epochs = self.epochs

        for k, v in model_params.items():
            if isinstance(v, torch.Tensor):
                model_params[k] = v.to(self._device)

        self._model.to(self._device)

        if training_dataset is not None:
            if y_train is not None or x_train is not None:
                raise RuntimeError('parameter <data_loader> is exclusive of <x_train> and <y_train>')
        else:
            if y_train is None or x_train is None:
                raise RuntimeError('missing parameter <x_train> or <y_train>')

        if training_dataset:
            for i_epoch in tqdm(range(self._total_iters, epochs + self._total_iters)):
                for i_batch, (x, y) in enumerate(training_dataset):

                    # convert to tuple for convenient
                    if not isinstance(x, tuple):
                        x = (x,)

                    # move tensor device
                    x = self._to_device(*x)
                    y = y.to(self._device)
                    self._model.train()

                    # feed model
                    def closure():
                        self.optimizer.zero_grad()
                        y_pred_ = self._model(*x, **model_params)
                        if isinstance(y_pred_, tuple):
                            model_params.update(y_pred_[1])
                            y_pred_ = y_pred_[0]
                        loss_ = self.loss_func(y_pred_, y)
                        loss_.backward()

                        if self.model_modifier is not None:
                            self.model_modifier(self._model)

                        return loss_

                    train_loss = self.optimizer.step(closure).item() / y.size(0)

                    step_info = OrderedDict(i_epoch=i_epoch + 1, i_batch=i_batch + 1, train_loss=train_loss)
                    self._exec_ext(step_info)
                    self._step_info.append(step_info)
                    self._total_iters = i_epoch + 1

                    if save_training_state:
                        self._model_states.append(self._model.state_dict())

                    yield step_info

        else:
            if not isinstance(x_train, tuple):
                x_train = (to_tensor(x_train),)
            else:
                x_train = tuple([to_tensor(x_) for x_ in x_train])

            y_train = to_tensor(y_train)

            # move tensor device
            x = self._to_device(*x_train)
            y = y_train.to(self._device)
            # import pdb;
            # pdb.set_trace()
            for i_epoch in tqdm(range(self._total_iters, epochs + self._total_iters)):

                self._model.train()

                # feed model
                def closure():
                    self.optimizer.zero_grad()
                    y_pred_ = self._model(*x, **model_params)
                    if isinstance(y_pred_, tuple):
                        model_params.update(y_pred_[1])
                        y_pred_ = y_pred_[0]
                    loss_ = self.loss_func(y_pred_, y)
                    loss_.backward()

                    if self.model_modifier is not None:
                        self.model_modifier(self._model, loss=loss_)

                    return loss_

                train_loss = self.optimizer.step(closure).item() / y.size(0)

                step_info = OrderedDict(i_epoch=i_epoch + 1, train_loss=train_loss)
                self._exec_ext(step_info)
                self._step_info.append(step_info)
                self._total_iters = i_epoch + 1

                if save_training_state:
                    self._model_states.append(self._model.state_dict())

                yield step_info

    def predict(self, x: Union[T_Data, Tuple[T_Data]], *, predict_only: bool = True) -> T_Prediction:
        """
        Predict from x input.
        This is just a simple wrapper for :meth:`~model.nn.utils.Predictor.predict`.

        Parameters
        ----------
        x: Union[T_Data, Tuple[T_Data]]
            Input data for prediction..
        predict_only: bool
            If ``False``, will returns all whatever the model returns.
            This can be useful for RNN models because these model also
            return `hidden variables` for recurrent training.

        Returns
        -------
        ret: T_Prediction
            Predict results.
        """
        return self._predictor(x, predict_only=predict_only)

    def _exec_ext(self, step_info):
        for ext in self._extensions:
            ext.run(step_info, self)

    def extend(self, ext: BaseExtension):
        self._extensions.append(ext)

    def as_dict(self):
        return OrderedDict(
            total_iteration=self._total_iters,
            losses=self._step_info,
            states=self._model_states,
            model=deepcopy(self._model.cpu())
        )
