# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from datetime import datetime as dt
from warnings import warn

import numpy as np
import torch as tc
import torch.nn as nn
from pandas import DataFrame as df
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.externals import joblib
from torch.autograd import Variable as Var

from .wrap import SL
from ... import get_conf
from ...utils.datatools import Saver


class Checker(object):
    """
    Check point.
    """

    def __init__(self, name):
        """

        Parameters
        ----------
        model: torch.nn.Module
            Something
        extra_para_list: dict
            Something
        """
        self.name = name
        self.saver = Saver(name, absolute=True, backend=SL())

    def __getitem__(self, item):
        if isinstance(item, int):
            return dict(model_state=self.saver['model_state', item],
                        epochs=self.saver['epochs', item],
                        y_pred=self.saver['y_pred', item],
                        loss=self.saver['loss', item])
        if isinstance(item, tuple):
            return self.saver.__getitem__(item)

        raise ValueError('except int or slice like [str, int]')

    def __call__(self, **kwargs):
        self.saver(**kwargs)


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
                 ctx='cpu',
                 check_step=100,
                 ignore_except=True,
                 log_step=0,
                 cache_path=None,
                 verbose=True
                 ):
        """

        Parameters
        ----------
        epochs: int
        log_step: int
        ctx: str
        check_step: int
        ignore_except: bool
        cache_path: str
        verbose: bool
            Print :class:`ModelRunner` environment.
        """
        self.verbose = verbose
        self.ignore_except = ignore_except
        self.check_step = check_step
        self.ctx = ctx
        self.epochs = epochs
        self.log_step = log_step
        self.cache_path = cache_path if cache_path else get_conf('usermodel')
        self.checker = None
        self.model = None
        self.loss_func = None
        self.optim = None
        self.lr = None
        self.lr_scheduler = None

    def __enter__(self):
        if self.verbose:
            print('Runner environment:')
            print('Epochs: {}'.format(self.epochs))
            print('Context: {}'.format(self.ctx.upper()))
            print('Ignore exception: {}'.format(self.ignore_except))
            print('Check step: {}'.format(self.check_step))
            print('Log step: {}\n'.format(self.log_step))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, model, name=None, *, loss_func=None, optim=None, lr=0.001, lr_scheduler=None):
        if isinstance(model, nn.Module):
            if not name:
                datetime = dt.now().strftime('-%Y-%m-%d_%H-%M-%S_%f')
                name = model.sig + datetime
            self.checker = Checker(self.cache_path + '/' + name)
            self.model = model
            self.loss_func = loss_func
            self.optim = optim
            self.lr = lr
            self.lr_scheduler = lr_scheduler
        else:
            raise ValueError(
                'Runner need a `torch.nn.Module` instance as first parameter but got {}'.format(type(model)))

    @staticmethod
    def _d2tv(data):
        if isinstance(data, df):
            data = tc.from_numpy(data.as_matrix()).type(tc.FloatTensor)
        elif isinstance(data, np.ndarray):
            data = tc.from_numpy(data).type(tc.FloatTensor)
        else:
            raise ValueError('need to be <numpy.ndarray> or <pandas.DataFrame> but got {}'.format(type(data)))
        return Var(data, requires_grad=False)

    def fit(self, x, y=None):
        """
        Fit Neural Network model

        Parameters
        ----------
        x: ``numpy.ndarray`` or ``pandas.DataFrame``
            Training data.
        y: ``numpy.ndarray`` or ``pandas.DataFrame``
            Target values.

        Returns
        -------
        self:
            returns an instance of self.
        """

        # transform to torch tensor
        x = self._d2tv(x)
        y = self._d2tv(y)

        # if use CUDA acc
        if self.ctx.lower() == 'gpu':
            if tc.cuda.is_available():
                self.model.cuda()
                x = x.cuda()
                y = y.cuda()
            else:
                warn('No cuda environment, use cpu fallback.', RuntimeWarning)
        else:
            self.model.cpu()

        # optimization
        optim = self.optim(self.model.parameters(), lr=self.lr)

        # adjust learning rate
        scheduler = None
        if self.lr_scheduler is not None:
            scheduler = self.lr_scheduler(optim)

        # train
        loss, y_pred = None, None
        try:
            print('=======start training=======')
            for t in range(self.epochs):
                if scheduler and not isinstance(scheduler, tc.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                y_pred = self.model(x)
                loss = self.loss_func(y_pred, y)
                if scheduler and isinstance(scheduler, tc.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()

                if self.log_step and t % self.log_step == 0:
                    print('at step: {}, Loss={:.4f}'.format(t, loss.data[0]))
                if self.check_step > 0 and t % self.check_step == 0:
                    self.checker(model_state=self.model.state_dict(),
                                 epochs=t,
                                 y_pred=y_pred.cpu().data.numpy(),
                                 loss=loss.data[0])

            print('=======over training=======')
            print('Loss={:.4f}\n'.format(loss.data[0]))
        except Exception as e:
            if self.ignore_except:
                pass
            else:
                raise e

        # save last results
        self.checker(model_state=self.model.state_dict(),
                     epochs=self.epochs,
                     y_pred=y_pred.cpu().data.numpy(),
                     loss=loss.data[0])
        return self

    def predict(self, x):
        # prepare x
        x = self._d2tv(x)

        # if use CUDA acc
        if self.ctx.lower() == 'gpu':
            if tc.cuda.is_available():
                self.model.cuda()
                x = x.cuda()
            else:
                warn('No cuda environment, use cpu fallback.', RuntimeWarning)
        else:
            self.model.cpu()

        # prediction
        pre_y = self.model(x)

        if self.ctx == 'GPU'.lower():
            return pre_y.cpu().data.numpy()
        return pre_y.data.numpy()

    def dump(self, fpath, **kwargs):
        val = dict(model=self.model, **kwargs)
        joblib.dump(val, fpath + '.pkl.z')
        pass
