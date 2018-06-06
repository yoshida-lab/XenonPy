# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
import types
from datetime import datetime, timedelta
from functools import wraps
from platform import version, system

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.base import BaseEstimator, RegressorMixin

import math
from .checker import Checker
from ..._conf import __version__
from ...utils import TimedMetaClass


def persist(*args, **kwargs):
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

    def _checked(o):
        if not isinstance(o, BaseRunner):
            raise TypeError('persistence only decorate <BaseRunner> inherent object\'s method')
        return o

    # only args or kwargs, no mix
    if (n_args == 0 and n_kwargs == 0) or (n_args != 0 and n_kwargs != 0):
        raise RuntimeError('Decorator need a function or method as first para')

    # if first para is func
    if n_args == 1:
        if isinstance(args[0], (types.FunctionType, types.MethodType)):
            fn = args[0]

            @wraps(fn)
            def _func(self, *args_, **kwargs_):
                self = _checked(self)
                ret = fn(self, *args_, **kwargs_)
                checker = getattr(self._checker, fn.__name__)
                checker.save(ret)
                return ret

            return _func

    # for name paras
    if n_args >= 1:
        if not all([isinstance(o, str) for o in args]):
            raise TypeError('Name of key must be str')

        def _deco(fn_):
            @wraps(fn_)
            def _func_1(self, *args_, **kwargs_):
                self = _checked(self)
                ret = fn_(self, *args_, **kwargs_)
                if not isinstance(ret, tuple):
                    ret = (ret,)
                _n_ret = len(ret)
                if _n_ret != n_args:
                    raise RuntimeError('Number of keys not equal values\' number')
                pair = zip(args, ret)
                checker = getattr(self._checker, fn_.__name__)
                checker.save(**{k: v for k, v in pair})
                return ret if len(ret) > 1 else ret[0]

            return _func_1

        return _deco

    if n_kwargs >= 1:
        if not all([isinstance(v, type) for v in kwargs.values()]):
            raise RuntimeError('Values must be type')

        def _deco(fn_):
            @wraps(fn_)
            def _func_2(self, *args_, **kwargs_):
                self = _checked(self)
                ret = fn_(self, *args_, **kwargs_)
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
                checker = getattr(self._checker, fn_.__name__)
                checker.save(**{k: v for k, v in pair})
                return ret if len(ret) > 1 else ret[0]

            return _func_2

        return _deco


class BaseRunner(BaseEstimator, metaclass=TimedMetaClass):

    def __init__(self, epochs=2000, *,
                 cuda: bool or int or str = False,
                 work_dir='.',
                 verbose=True):
        self._epochs = epochs
        if isinstance(cuda, bool):
            self._device = torch.device('cuda') if cuda else torch.device('cpu')
        elif isinstance(cuda, int):
            self._device = torch.device('cuda', cuda)
        else:
            self._device = torch.device(cuda)
        self._verbose = verbose
        self._work_dir = work_dir
        self._model = None
        self._model_name = None
        self._checker = None
        self._logs = []

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.stop()
        elapsed = str(timedelta(seconds=self.elapsed))
        self.logger('done runner <%s>: %s' % (self.__class__.__name__, datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        self.logger('total elapsed time: %s' % elapsed)
        logs = '\n'.join(self._logs)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        with open(self.work_dir + '/log_' + now + '.txt', 'w') as f:
            f.write(logs)

    def __enter__(self):
        self.logger('start runner <%s> at %s' % (self.__class__.__name__, datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        self._timer.start()
        return self

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
    def device(self):
        return self._device

    @device.setter
    def device(self, v):
        if not isinstance(v, torch.torch.device):
            raise TypeError('Need torch.device object')
        self._device = v

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
        self._checker.save(runner=dict(epochs=self._epochs,
                                       verbose=self._verbose,
                                       work_dir=self._work_dir))

        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def persist(self, *args, **kwargs):
        """
        Persist data.
        This is a wrap of ::class:`Dataset`

        Parameters
        ----------
        args
        kwargs
        """
        self._checker.save(*args, **kwargs)

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
    def tensor(*data_and_type):
        def _tensor(data, torch_type):
            if isinstance(data, pd.DataFrame):
                return torch.from_numpy(data.as_matrix()).to(torch_type)
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(torch_type)
            else:
                raise TypeError('Need <numpy.ndarray> or <pandas.DataFrame> but got %s' % type(data))

        return tuple([_tensor(data_, type_) for data_, type_ in data_and_type])

    def to_device(self, *tensor):
        # if use CUDA acc
        if self._device.type != 'cpu':
            return tuple([t.cuda(self._device, True) for t in tensor])
        return tuple([t.cpu() for t in tensor])

    def batch_tensor(self, *data,
                     batch_size=0.2,
                     shuffle=True,
                     num_worker=0,
                     pin_memory=True):
        # batch_size
        if not data:
            return None
        if not all([isinstance(o, (pd.DataFrame, np.ndarray)) for o, _ in data]):
            raise TypeError('Need <numpy.ndarray> or <pandas.DataFrame>')
        if isinstance(batch_size, float):
            batch_size = math.floor(data[0][0].shape[0] * batch_size)

        return Data.DataLoader(dataset=Data.TensorDataset(*self.to_device(*self.tensor(*data))),
                               batch_size=batch_size,
                               shuffle=shuffle,
                               num_workers=num_worker,
                               pin_memory=pin_memory)

    def logger(self, *info):
        log = '|> ' + '\n|> '.join(info)
        self._logs.append(log)
        if self._verbose:
            print(log)

    def fit(self, x_train=None, y_train=None, *,
            data_loader=None,
            x_dtype=torch.float,
            y_dtype=torch.float,
            describe=None):
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
        x_dtype: tensor type
            Corresponding dtype in torch tensor. Default is torch.float.
            Detials: https://pytorch.org/docs/stable/tensors.html
        y_dtype: tensor types
            Corresponding dtype in torch tensor. Default is torch.float.
        describe: dict
            Additional information to describe this model training.

        Returns
        -------
        any
            returns ::meth:`optim` results.
        """

        def _ite():
            self._model.to(self._device)
            if not data_loader:
                x_, y_ = self.to_device(*self.tensor((x_train, x_dtype), (y_train, y_dtype)))
                for t in range(self._epochs):
                    yield y_, self._model(x_), t, None
                return

            if data_loader:
                for t in range(self._epochs):
                    for i, (x_, y_) in enumerate(data_loader):
                        x_, y_ = self.to_device(self._device, x_, y_)
                        yield y_, self._model(x_), t, i
                return

        desc = dict(
            python=version(),
            system=system(),
            numpy=np.__version__,
            torch=torch.__version__,
            xenonpy=__version__,
            structure=str(self._model),
            running_at=self._work_dir,
            start=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
        )

        # training
        now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        self.logger('training model: <%s>' % self._model_name)
        self.logger('start: %s' % now, '')
        start = self.elapsed

        self._model.train(True)
        ret = self.optim(_ite())  # user implementation
        self._model.train(False)

        now = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        elapsed = str(timedelta(seconds=self.elapsed - start))
        self.logger('done: %s' % now)
        self.logger('elapsed time: %s\n' % elapsed)

        if describe and isinstance(describe, dict):
            self._checker.save(describe={**desc, **describe, 'done': now})
        else:
            self._checker.save(describe={**desc, 'done': now})
        self._checker.trained_model = self._model

        return ret

    def predict(self, x_test, y_test, *,
                x_dtype=torch.float,
                y_dtype=torch.float):
        """

        Parameters
        ----------
        x_test: DataFrame, ndarray
            Input data for test..
        y_test: DataFrame, ndarray
            Target data for test.
        x_dtype: tensor type
            Corresponding dtype in torch tensor. Default is torch.float.
            Detials: https://pytorch.org/docs/stable/tensors.html
        y_dtype: tensor types
            Corresponding dtype in torch tensor. Default is torch.float.

        Returns
        -------
        any
            Return ::meth:`post_predict` results.
        """
        # prepare data
        x_test, y_test = self.to_device(*self.tensor((x_test, x_dtype), (y_test, y_dtype)))

        # prediction
        self._model.to(self._device)
        y_true, y_pred = self._model(x_test), y_test
        return self.post_predict(y_true, y_pred)

    @classmethod
    def from_checker(cls, checker, checkpoint=None):
        runner = checker.last('runner')
        ret = cls()
        ret._verbose = runner['verbose']
        ret._epochs = runner['epochs']
        ret._work_dir = runner['work_dir']
        ret._checker = checker
        ret._model_name = checker.model_name
        if not checkpoint:
            ret._model = checker.trained_model if checker.trained_model else checker.init_model
        else:
            model_state, _ = checker[checkpoint]
            ret._model = checker.init_model.load_state_dict(model_state)
        return ret


class RegressionRunner(BaseRunner, RegressorMixin):
    """
    Run model.
    """

    def __init__(self, epochs=2000, *,
                 cuda=False,
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
        cuda: str
        check_step: int
        work_dir: str
        verbose: bool
            Print :class:`ModelRunner` environment.
        """
        super(RegressionRunner, self).__init__(epochs, cuda=cuda, work_dir=work_dir, verbose=verbose)
        self._check_step = check_step
        self._log_step = log_step
        self._lr = 0.01
        self._lr_scheduler = None

        self.logger('Runner environment:',
                    'Running dir: {}'.format(self._work_dir),
                    'Epochs: {}'.format(self._epochs),
                    'Context: {}'.format(self._device),
                    'Check step: {}'.format(self._check_step),
                    'Log step: {}\n'.format(self._log_step))

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

    @persist('y_true', 'y_pred')
    def post_predict(self, y_true, y_pred):
        return y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()

    def optim(self, iter_):
        # optimization
        optim = torch.optim.Adam(self._model.parameters(), lr=self._lr)

        # adjust learning rate
        scheduler = self._lr_scheduler(optim) if self._lr_scheduler else None
        y_train = y_pred = loss = None
        start = self.elapsed
        for y_train, y_pred, t, i in iter_:

            if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            loss = F.mse_loss(y_pred, y_train)
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self._log_step > 0 and t % self._log_step == 0:
                elapsed = str(timedelta(seconds=self.elapsed - start))
                start = self.elapsed
                self.logger('{}/{}, Loss={:.4f}, elapsed time: {}'.format(t, self._epochs, loss, elapsed))
            if self._check_step > 0 and t % self._check_step == 0:
                self.checkpoint(mse_loss=loss.item())

        self.logger('Final loss={:.4f}'.format(loss), '')

        return y_train.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
