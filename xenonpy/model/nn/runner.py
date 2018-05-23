# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
import math
import types
from datetime import datetime as dt
from functools import wraps
from platform import version, system
from warnings import warn

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from pandas import DataFrame as df
from sklearn.base import BaseEstimator
from sklearn.externals import joblib

from .checker import Checker
from ... import __version__
from ...utils.functional import TimedMetaClass


class BaseRunner(BaseEstimator, metaclass=TimedMetaClass):

    def __init__(self, epochs=2000, ctx='cpu', work_dir='.'):
        self._epochs = epochs
        self._ctx = ctx
        self._work_dir = work_dir
        self._models = None
        self._model_name = None
        self._checker = None

    def optim(self, iter_):
        """

        Parameters
        ----------
        iter_: generator
            Yields y_train, self._model(x_train), n_ite, n_batch

        Returns
        -------
        any

        """
        raise NotImplementedError

    def post_predict(self, y_true, y_pred):
        """

        Parameters
        ----------
        y_true: torch.Tensor
        y_pred: torch.Tensor

        Returns
        -------
        any
        """
        raise NotImplementedError

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, v):
        self._epochs = v

    @property
    def ctx(self):
        return self._ctx

    @ctx.setter
    def ctx(self, v):
        self._ctx = v

    @property
    def work_dir(self):
        return self._work_dir

    @work_dir.setter
    def work_dir(self, v):
        self._work_dir = v

    @property
    def elapsed(self):
        return self._timer.elapsed

    def __call__(self, model, name=None, **kwargs):
        """"""
        # model must inherit form nn.Module
        if not isinstance(model, nn.Module):
            raise TypeError('Need <torch.nn.Module> instance but got %s' % type(model))

        if not name:
            from hashlib import md5
            sig = md5(model.__str__().encode()).hexdigest()
            name = sig
        self._checker = Checker(name, self._work_dir)
        self._model = model
        self._model_name = name
        self._checker.init_model = model
        self._checker.save(runner=dict(ctx=self._ctx,
                                       epochs=self._epochs,
                                       work_dir=self._work_dir))

        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k + '_', v)

    def persistence(*args, **kwargs):
        """

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        n_args = len(args)
        n_kwargs = len(kwargs)

        # only args or kwargs, no mix
        if (n_args == 0 and n_kwargs == 0) or (n_args != 0 and n_kwargs != 0):
            raise RuntimeError('Decorator need a function or method as first para')

        # if first para is func
        if n_args == 1:
            arg = args[0]
            if isinstance(arg, (types.FunctionType, types.MethodType)):
                @wraps
                def _func(self, *args_, **kwargs_):
                    ret = arg(self, *args_, **kwargs_)
                    self._checker.save(ret)
                    return ret

                return _func

        # for name paras
        if n_args >= 1:
            if not all([isinstance(o, str) for o in args]):
                raise TypeError('Name of key must be str')

            def _deco(fn):
                @wraps
                def _func_1(self, *args_, **kwargs_):
                    ret = fn(self, *args_, **kwargs_)
                    if not isinstance(ret, tuple):
                        ret = (ret,)
                    _n_ret = len(ret)
                    if _n_ret != n_args:
                        raise RuntimeError('Number of keys not equal values\' number')
                    pair = zip(args, ret)
                    self._checker.save(**{k: v for k, v in pair})
                    return ret if len(ret) > 1 else ret[0]

                return _func_1

            return _deco

        if n_kwargs >= 1:
            def _deco(fn):
                @wraps
                def _func_2(self, *args_, **kwargs_):
                    ret = fn(self, *args_, **kwargs_)
                    if not isinstance(ret, tuple):
                        ret = (ret,)
                    _n_ret = len(ret)
                    if _n_ret != n_kwargs:
                        raise RuntimeError('Number of keys not equal values\' number')
                    types_ = kwargs.values()
                    if not all([isinstance(v, t) for v, t in zip(ret, types_)]):
                        raise TypeError('Returns\' type not match')
                    names = kwargs.keys()
                    pair = zip(names, ret)
                    self._checker.save(**{k: v for k, v in pair})
                    return ret if len(ret) > 1 else ret[0]

                return _func_2

            return _deco

    def checkpoint(self, **describe):
        """
        Take a snapshot for current model status.

        Parameters
        ----------
        describe: dict
            Additional description for checkpoint.

        """
        self._checker(model_state=self._model.state_dict(), **describe)

    @staticmethod
    def _to_tensor(*data_type):
        def _tensor(data, torch_type):
            if isinstance(data, df):
                return torch.from_numpy(data.as_matrix()).type(torch_type)
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).type(torch_type)
            else:
                raise TypeError('Need <numpy.ndarray> or <pandas.DataFrame> but got %s' % type(data))

        return tuple([_tensor(data_, type_) for data_, type_ in data_type])

    def _cuda(self, *tensor):
        # if use CUDA acc
        if self._ctx.lower() == 'gpu':
            if torch.cuda.is_available():
                self._model.cuda()
                return tuple([t.cuda() for t in tensor])
            else:
                warn('No cuda environment, use cpu fallback.', RuntimeWarning)
        else:
            self._model.cpu()
            return tuple([t.cpu() for t in tensor])

    def fit(self, x_train, y_train, *,
            x_torch_type=torch.FloatTensor,
            y_torch_type=torch.FloatTensor,
            describe=None,
            batch_size=None,
            shuffle=False,
            num_worker=0,
            pin_memory=False):
        """
        Fit Neural Network model

        Parameters
        ----------
        x_train:  ``numpy.ndarray`` or ``pandas.DataFrame``.
            Training data.
        y_train:  ``numpy.ndarray`` or ``pandas.DataFrame``.
            Test data.
        x_torch_type: torch data type
            Default is torch.FloatTensor.
        y_torch_type: torch data type
            Default is torch.FloatTensor.
        describe: dict
            Additional information to describe this model training.
        batch_size: int, float
        pin_memory: bool
            If ``True``, the data loader will copy tensors into CUDA pinned memory before returning them.
        num_worker: int
            How many subprocesses to use for data loading.
            0 means that the data will be loaded in the main process. (default: 0)
        shuffle: bool
            Set to ``True`` to have the data reshuffled at every epoch (default: False).
        Returns
        -------
        self:
            returns an instance of self.
        """

        # prepare data
        x_train, y_train = self._to_tensor((x_train, x_torch_type), (y_train, y_torch_type))
        x_train, y_train = self._cuda(x_train, y_train)

        # batch_size
        if batch_size:
            if isinstance(batch_size, int):
                if batch_size > x_train.shape[0]:
                    warn('Batch size %d is greater than sample size, set batch size to %d.'
                         % (batch_size, x_train.shape[0]), RuntimeWarning)
                    batch_size = x_train.shape[0]
            elif isinstance(batch_size, float):
                batch_size = math.floor(x_train.shape[0] * batch_size)
        else:
            batch_size = x_train.shape[0]

        train_loader = Data.DataLoader(dataset=Data.TensorDataset(x_train, y_train),
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_worker,
                                       pin_memory=pin_memory)

        def _ite():
            for t in range(self._epochs):
                for i, (x_, y_) in enumerate(train_loader):
                    yield y_, self._model(x_), t, i

        desc = dict(
            python=version(),
            system=system(),
            numpy=np.__version__,
            torch=torch.__version__,
            xenonpy=__version__,
            structure=str(self._model),
            running_at=self._work_dir,
            start=dt.now().strftime('%Y-%m-%d_%H-%M-%S_%f'),
        )

        # training
        print('|> training start ===> ', dt.now().strftime('%Y-%m-%d_%H-%M-%S'))
        print('|> Model name: {}\n'.format(self._model_name))

        ret = self.optim(_ite())  # user implementation

        print('\n|> elapsed time: {}'.format(self.elapsed))
        print('|> training done ===> ', dt.now().strftime('%Y-%m-%d_%H-%M-%S'))

        if describe and isinstance(describe, dict):
            self._checker.save(describe={**desc, **describe, 'done': dt.now().strftime('%Y-%m-%d_%H-%M-%S_%f')})
        else:
            self._checker.save(describe={**desc, 'done': dt.now().strftime('%Y-%m-%d_%H-%M-%S_%f')})
        self._checker.trained_model = self._model

        return ret

    def predict(self, x_test, y_test, *,
                x_torch_type=torch.FloatTensor,
                y_torch_type=torch.FloatTensor):
        """

        Parameters
        ----------
        x_test: DataFrame, ndarray
            Input data for test..
        y_test: DataFrame, ndarray
            Target data for test.
        x_torch_type: torch.Tensor
            Corresponding dtype in torch tensor.
        y_torch_type: torch.Tensor
            Corresponding dtype in torch tensor

        Returns
        -------
        any
            Return ::meth:`post_predict` results.
        """
        # prepare data
        x_test, y_test = self._to_tensor((x_test, x_torch_type), (y_test, y_torch_type))
        x_test, y_test = self._cuda(x_test, y_test)

        # prediction
        y_true, y_pred = self._model(x_test), y_test
        return self.post_predict(y_true, y_pred)

    @classmethod
    def from_checker(cls, name, path=None):
        checker = Checker.load(name, path)
        runner = checker.last('runner')
        ret = cls()
        ret._add_info = runner['add_info']
        ret._verbose = runner['verbose']
        ret._check_step = runner['check_step']
        ret._ctx = runner['ctx']
        ret._epochs = runner['epochs']
        ret._log_step = runner['log_step']
        ret._work_dir = runner['work_dir']
        ret._model_name = runner['model_name']
        ret._loss_func = runner['loss_func']
        ret._optim = runner['optim']
        ret._lr = runner['lr']
        ret._lr_scheduler = runner['lr_scheduler']
        try:
            ret._x_scaler = checker.last('x_scaler')
        except FileExistsError:
            pass
        try:
            ret._y_scaler = checker.last('y_scaler')
        except FileExistsError:
            pass
        try:
            ret._splitter = checker.last('splitter')
        except FileExistsError:
            pass
        ret._checker = checker
        ret._model = checker.trained_model if checker.trained_model else checker.init_model
        return ret

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
        val = dict(model=self._model, **kwargs)
        joblib.dump(val, fpath)


class RegressionRunner(BaseRunner):
    """
    Run model.
    """

    def __init__(self, epochs=2000, *,
                 ctx='cpu',
                 check_step=100,
                 log_step=0,
                 work_dir=None,
                 verbose=True
                 ):
        """

        Parameters
        ----------
        epochs: int
        log_step: int
        ctx: str
        check_step: int
        work_dir: str
        verbose: bool
            Print :class:`ModelRunner` environment.
        """
        super(RegressionRunner, self).__init__(epochs, ctx, work_dir)
        self._verbose = verbose
        self._check_step = check_step
        self._log_step = log_step
        self._lr = 0.01
        self._lr_scheduler = None

        if verbose:
            print('Runner environment:')
            print('Running dir: {}'.format(self._work_dir))
            print('Epochs: {}'.format(self._epochs))
            print('Context: {}'.format(self._ctx.upper()))
            print('Check step: {}'.format(self._check_step))
            print('Log step: {}\n'.format(self._log_step))

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, v):
        """"""
        self._lr = v

    @property
    def lr_scheduler(self):
        return None

    @lr_scheduler.setter
    def lr_scheduler(self, v):
        self._lr_scheduler = v

    @persistence('y_true', 'y_pred')
    def post_predict(self, y_true, y_pred):
        return y_true.cpu().numpy(), y_pred.cpu().numpy()

    def optim(self, train_loader):
        # optimization
        optim = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        # adjust learning rate
        scheduler = self._lr_scheduler(optim) if self._lr_scheduler else None

        y_train = y_pred = loss = None
        for y_train, y_pred, t, i in range(self._epochs):

            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            loss = F.mse_loss(y_pred, y_train)
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self._log_step > 0 and t % self._log_step == 0:
                print('|> {}/{}, Loss={:.4f}, elapsed time: {}'.format(t, self._epochs, loss, self.elapsed))
            if self._check_step > 0 and t % self._check_step == 0:
                self.checkpoint(mse_loss=loss.item())

        print('Final loss={:.4f}\n'.format(loss))

        return y_train.cpu().numpy(), y_pred.cpu().numpy()
