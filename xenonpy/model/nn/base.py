# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from functools import partial
from warnings import warn

import torch as tc
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Variable as Var


class Layer1d(nn.Module):
    def __init__(self, n_in: int, n_out: int, *,
                 p_drop=0.0,
                 layer_func=nn.Linear,
                 act_func=nn.ReLU(),
                 batch_normalize=False,
                 momentum=0.1,
                 lr=1.0
                 ):
        super().__init__()
        self.lr = lr
        self.neuron = layer_func(n_in, n_out)
        self.batch_nor = None if not batch_normalize else nn.BatchNorm1d(n_out, momentum)
        self.act_func = None if not act_func else act_func
        self.dropout = None if p_drop == 0.0 else nn.Dropout(p_drop)

    def forward(self, x):
        _out = self.neuron(x)
        if self.dropout:
            _out = self.dropout(_out)
        if self.batch_nor:
            _out = self.batch_nor(_out)
        if self.act_func:
            _out = self.act_func(_out)
        return _out


class ModelRunner(BaseEstimator, RegressorMixin):
    """
    Run model.
    """

    def __init__(self, epochs=2000, *,
                 verbose=0,
                 ctx='cpu',
                 check_step=100,
                 ignore_except=True
                 ):
        """

        Parameters
        ----------
        epochs: int
        verbose: int
        ctx: str
        check_step: int
        ignore_except: bool
        """
        self.ignore_except = ignore_except
        self.check_step = check_step
        self.ctx = ctx
        self.epochs = epochs
        self.verbose = verbose
        self.model = None
        self.loss_func = None
        self.optim = None
        self.lr = None
        self.lr_scheduler = None

    @classmethod
    def SGD(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.SGD`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.SGD
        """
        return partial(tc.optim.SGD, *args, **kwargs)

    @classmethod
    def Adadelta(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adadelta`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adadelta
        """
        return partial(tc.optim.Adadelta, *args, **kwargs)

    @classmethod
    def Adagrad(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adagrad`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adagrad
        """
        return partial(tc.optim.Adagrad, *args, **kwargs)

    @classmethod
    def Adam(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adam`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adam
        """
        return partial(tc.optim.Adam, *args, **kwargs)

    @classmethod
    def SparseAdam(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.SparseAdam`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.SparseAdam
        """
        return partial(tc.optim.SparseAdam, *args, **kwargs)

    @classmethod
    def Adamax(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adamax`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adamax
        """
        return partial(tc.optim.Adamax, *args, **kwargs)

    @classmethod
    def ASGD(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.ASGD`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.ASGD
        """
        return partial(tc.optim.ASGD, *args, **kwargs)

    @classmethod
    def LBFGS(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.LBFGS`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.LBFGS
        """
        return partial(tc.optim.LBFGS, *args, **kwargs)

    @classmethod
    def RMSprop(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.RMSprop`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.RMSprop
        """
        return partial(tc.optim.RMSprop, *args, **kwargs)

    @classmethod
    def Rprop(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Rprop`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Rprop
        """
        return partial(tc.optim.Rprop, *args, **kwargs)

    @classmethod
    def LambdaLR(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.LambdaLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.LambdaLR
        """
        return partial(tc.optim.lr_scheduler.LambdaLR, *args, **kwargs)

    @classmethod
    def StepLR(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.StepLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.StepLR
        """
        return partial(tc.optim.lr_scheduler.StepLR, *args, **kwargs)

    @classmethod
    def MultiStepLR(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.MultiStepLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.MultiStepLR
        """
        return partial(tc.optim.lr_scheduler.MultiStepLR, *args, **kwargs)

    @classmethod
    def ExponentialLR(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.ExponentialLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.ExponentialLR
        """
        return partial(tc.optim.lr_scheduler.ExponentialLR, *args, **kwargs)

    @classmethod
    def ReduceLROnPlateau(cls, *args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        return partial(tc.optim.lr_scheduler.ReduceLROnPlateau, *args, **kwargs)

    def __enter__(self):
        if self.verbose:
            print('Runner environment:')
            print('Epochs: {}'.format(self.epochs))
            print('Context: {}'.format(self.ctx))
            print('Check step: {}'.format(self.check_step))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, model, *, loss_func=None, optim=None, lr=0.001, lr_scheduler=None):
        if isinstance(model, nn.Module):
            self.model = model
            self.loss_func = loss_func
            self.optim = optim
            self.lr = lr
            self.lr_scheduler = lr_scheduler
        else:
            raise ValueError(
                'Runner need a `torch.nn.Module` instance as first parameter but got {}'.format(type(model)))

    def fit(self, x, y=None):
        """

        Parameters
        ----------
        x: numpy.ndarray or pandas.DataFrame
            X train.
        y: numpy.ndarray or pandas.DataFrame
            y property.
        Returns
        -------

        """

        # transform to torch tensor
        if not isinstance(x, Var):
            x = Var(x, requires_grad=False)
        _, col = x.size()
        if not isinstance(y, Var):
            y = Var(y, requires_grad=False)

        # if use CUDA acc
        if self.ctx == 'GPU':
            if tc.cuda.is_available():
                self.model.cuda()
                x = x.cuda()
                y = y.cuda()
            else:
                warn('No cuda environment, use cup fallback.', RuntimeWarning)

        # optimization
        optim = self.optim(self.model.parameters(), lr=self.lr)

        # adjust learning rate
        scheduler = None
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optim)

        # train
        loss = None
        for t in range(self.epochs):
            if scheduler:
                scheduler.setup()
            pre_y = self.model(x)
            loss = self.loss_func(pre_y, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose and t % self.verbose == 0:
                print('at step: {}, Loss={:.4f}'.format(t, loss.data[0]))
            # if self.check_step > 0 and t % self.check_step == 0:
            #     self.checkpoint(epochs=t, loss=loss.data[0])

        # self.trained = True
        # if self.check_step > 0:
        #     self.checkpoint.save(self.save_snap, self.save_model)
        print('Loss={:.4f}'.format(loss.data[0]))
        return self

    def predict(self, x, y=None):
        if not isinstance(x, Var):
            x = Var(x)
        pre_y = self.model(x)
        if y:
            if not isinstance(y, Var):
                y = Var(y)
            print(self.loss_func(pre_y, y))
        return pre_y

    def save(self, fpath, **kwargs):
        pass
