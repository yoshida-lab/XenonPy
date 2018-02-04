# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from functools import partial
from pathlib import Path
from warnings import warn

import numpy as np
import torch as tc
import torch.nn as nn
from pandas import DataFrame as df
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.externals import joblib
from torch.autograd import Variable as Var

from ... import get_conf
from ...utils.datatools import Saver


class _SL(object):
    def __init__(self):
        self.load = tc.load
        self.dump = tc.save


class Optim(object):
    @staticmethod
    def SGD(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.SGD`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.SGD
        """
        return partial(tc.optim.SGD, *args, **kwargs)

    @staticmethod
    def Adadelta(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adadelta`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adadelta
        """
        return partial(tc.optim.Adadelta, *args, **kwargs)

    @staticmethod
    def Adagrad(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adagrad`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adagrad
        """
        return partial(tc.optim.Adagrad, *args, **kwargs)

    @staticmethod
    def Adam(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adam`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adam
        """
        return partial(tc.optim.Adam, *args, **kwargs)

    @staticmethod
    def SparseAdam(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.SparseAdam`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.SparseAdam
        """
        return partial(tc.optim.SparseAdam, *args, **kwargs)

    @staticmethod
    def Adamax(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adamax`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adamax
        """
        return partial(tc.optim.Adamax, *args, **kwargs)

    @staticmethod
    def ASGD(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.ASGD`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.ASGD
        """
        return partial(tc.optim.ASGD, *args, **kwargs)

    @staticmethod
    def LBFGS(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.LBFGS`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.LBFGS
        """
        return partial(tc.optim.LBFGS, *args, **kwargs)

    @staticmethod
    def RMSprop(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.RMSprop`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.RMSprop
        """
        return partial(tc.optim.RMSprop, *args, **kwargs)

    @staticmethod
    def Rprop(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Rprop`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Rprop
        """
        return partial(tc.optim.Rprop, *args, **kwargs)


class LrScheduler(object):
    @staticmethod
    def LambdaLR(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.LambdaLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.LambdaLR
        """
        return partial(tc.optim.lr_scheduler.LambdaLR, *args, **kwargs)

    @staticmethod
    def StepLR(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.StepLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.StepLR
        """
        return partial(tc.optim.lr_scheduler.StepLR, *args, **kwargs)

    @staticmethod
    def MultiStepLR(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.MultiStepLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.MultiStepLR
        """
        return partial(tc.optim.lr_scheduler.MultiStepLR, *args, **kwargs)

    @staticmethod
    def ExponentialLR(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.ExponentialLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.ExponentialLR
        """
        return partial(tc.optim.lr_scheduler.ExponentialLR, *args, **kwargs)

    @staticmethod
    def ReduceLROnPlateau(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        return partial(tc.optim.lr_scheduler.ReduceLROnPlateau, *args, **kwargs)


class Checker(object):
    """
    Check point.
    """

    def __init__(self, fpath):
        """

        Parameters
        ----------
        model: torch.nn.Module
            Something
        extra_para_list: dict
            Something
        """
        i = 1
        while Path(fpath + '@' + str(i)).expanduser().exists():
            i += 1
        fpath = fpath + '@' + str(i)
        self.name = Path(fpath).stem
        self.saver = Saver(fpath, absolute=True)
        self.saver.pkl = _SL()

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


class ModelRunner(BaseEstimator, RegressorMixin):
    """
    Run model.
    """

    def __init__(self, epochs=2000, *,
                 ctx='cpu',
                 check_step=100,
                 ignore_except=True,
                 log_step=0,
                 execute_dir=None,
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
        execute_dir: str
        verbose: bool
            Print :class:`ModelRunner` environment.
        """
        self.verbose = verbose
        self.ignore_except = ignore_except
        self.check_step = check_step
        self.ctx = ctx
        self.epochs = epochs
        self.log_step = log_step
        self.execute_dir = execute_dir if execute_dir else get_conf('usermodel')
        self.checker = None
        self.model = None
        self.loss_func = None
        self.optim = None
        self.lr = None
        self.lr_scheduler = None

    def __enter__(self):
        if self.verbose:
            print('Runner environment:')
            print('Running dir: {}'.format(self.execute_dir))
            print('Epochs: {}'.format(self.epochs))
            print('Context: {}'.format(self.ctx.upper()))
            print('Ignore exception: {}'.format(self.ignore_except))
            print('Check step: {}'.format(self.check_step))
            print('Log step: {}\n'.format(self.log_step))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __call__(self, model, name=None, *, loss_func=None, optim=None, lr=0.001, lr_scheduler=None):
        if isinstance(model, nn.Module):
            if not name:
                name = model.sig
            self.checker = Checker(self.execute_dir + '/' + name)
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
                self.ctx = 'CPU'
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
                self.ctx = 'CPU'
                warn('No cuda environment, use cpu fallback.', RuntimeWarning)
        else:
            self.model.cpu()

        # prediction
        try:
            pre_y = self.model(x)

            if self.ctx.lower() == 'gpu':
                return pre_y.cpu().data.numpy()
            return pre_y.data.numpy()
        except Exception as e:
            if self.ignore_except:
                pass
            else:
                raise e

    def from_checkpoint(self, fname):
        raise NotImplementedError('Not implemented')

    def dump(self, fpath, **kwargs):
        """
        Save model into pickled file with at ``fpath``.
        Some additional description can be given from ``kwargs``.

        Parameters
        ----------
        fpath: str
            Path with name of pickle file.
        kwargs: dict
            Additional description
        """
        val = dict(model=self.model, **kwargs)
        joblib.dump(val, fpath)
