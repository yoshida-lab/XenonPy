#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import namedtuple
from typing import Union

import torch
from sklearn.base import BaseEstimator
from torch.nn import Module

from .wrap.base import BaseOptimizer, BaseLRScheduler
from ...utils import TimedMetaClass


class BaseRunner(BaseEstimator, metaclass=TimedMetaClass):

    def __init__(self,
                 model: torch.nn.Module,
                 loss: Module,
                 optimizer: BaseOptimizer,
                 *,
                 lr_scheduler: BaseLRScheduler = None,
                 epochs: int = 2000,
                 cuda: Union[bool, str] = False,
                 verbose: bool = True,
                 ):
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

    @staticmethod
    def _check_cuda(cuda):
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
            x_train=None,
            y_train=None,
            *,
            data_loader=None,
            yield_: str = 'none',
            model_params: dict = None):
        """
        Fit Neural Network model

        Parameters
        ----------
        x_train:  ``numpy.ndarray`` or ``pandas.DataFrame``.
            Training data. Will be ignored will``data_loader`` is given.
        y_train:  ``numpy.ndarray`` or ``pandas.DataFrame``.
            Test data. Will be ignored will``data_loader`` is given.
        data_loader: torch.data.DataLoader
            Torch DataLoader. If given, will only use this as training dataset.
        yield_ : str
            Yields intermediate information.
        model_params: dict
            Other model parameters.

        Yields
        ------
        namedtuple

        """
        if yield_ not in ['loss', 'none', 'all']:
            raise RuntimeError(f'<yield_> can only be "loss", "all", and "none" but got {yield_}')

        if model_params is None:
            model_params = {}

        self._model.to(self._device)
        self._model.train()

        yields_all = namedtuple('yields', 'y_pred y_true loss i_epoch i_batch model_params')
        yields_loss = namedtuple('yields', 'loss i_epoch i_batch')

        if data_loader is not None:
            if y_train is not None or x_train is not None:
                raise RuntimeError('parameter <data_loader> is exclusive of <x_train> and <y_train>')
        else:
            if y_train is None or x_train is None:
                raise RuntimeError('missing parameter <x_train> or <y_train>')

        for i_epoch in range(self.epochs):
            if data_loader:
                for i_batch, (x_, y_) in enumerate(data_loader):

                    # convert to tuple for convenient
                    if not isinstance(x_, tuple):
                        x_ = (x_,)
                    if not isinstance(y_, tuple):
                        y_ = (y_,)

                    # move tensor device
                    x_ = self.to_device(*x_)
                    y_ = self.to_device(*y_)
                    if len(y_) == 1:
                        y_ = y_[0]

                    # feed model
                    def closure():
                        self.optimizer.zero_grad()
                        y_pred = self._model(*x_, **model_params)
                        if isinstance(y_pred, tuple):
                            y_pred = y_pred[0]
                            model_params.update(y_pred[1])
                        loss = self.loss_func(y_pred, y_)
                        loss.backward()
                        if yield_ == 'all':
                            yield yields_all(y_pred=y_pred, y_true=y_, loss=loss / y_.size(0), i_epoch=i_epoch,
                                             i_batch=i_batch, model_params=model_params)
                        return loss

                    loss = self.optimizer.step(closure)
                    if yield_ == 'loss':
                        yield yields_loss(loss=loss / y_.size(0), i_epoch=i_epoch, i_batch=i_batch)

            else:
                if not isinstance(x_train, tuple):
                    x_train = (x_train,)
                if not isinstance(y_train, tuple):
                    y_train = (y_train,)

                # move tensor device
                x_ = self.to_device(*x_train)
                y_ = self.to_device(*y_train)
                if len(y_) == 1:
                    y_ = y_[0]

                # feed model
                def closure():
                    self.optimizer.zero_grad()
                    y_pred = self._model(*x_, **model_params)
                    if isinstance(y_pred, tuple):
                        y_pred = y_pred[0]
                        model_params.update(y_pred[1])
                    loss = self.loss_func(y_pred, y_)
                    loss.backward()
                    if yield_ == 'all':
                        yield yields_all(y_pred=y_pred, y_true=y_, loss=loss / y_.size(0), i_epoch=i_epoch,
                                         i_batch=i_batch, model_params=model_params)
                    return loss

                loss = self.optimizer.step(closure)
                if yield_ == 'loss':
                    yield yields_loss(loss=loss / y_.size(0), i_epoch=i_epoch, i_batch=i_batch)

    def to_device(self, *tensor):
        # if use CUDA acc
        if self._device.type != 'cpu':
            return tuple([t.cuda(self._device, True) for t in tensor])
        return tuple([t.cpu() for t in tensor])

    def predict(self, x_test, *, to_cpu=True):
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
        x_test, = self.to_device(*x_test)

        # prediction
        self._model.to(self._device)
        self._model.eval()

        y_pred = self._model(*x_test).detach()
        if to_cpu:
            return y_pred.cpu().numpy()
        return y_pred
