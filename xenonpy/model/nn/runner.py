#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
import types
from copy import deepcopy
from datetime import datetime, timedelta
from functools import wraps
from hashlib import md5
from pathlib import Path
from platform import version as sys_ver
from sys import version as py_ver

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

from .base import BaseRunner
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
                checker = getattr(self.checker, fn.__name__)
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
                checker = getattr(self.checker, fn_.__name__)
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
                checker = getattr(self.checker, fn_.__name__)
                checker.save(**{k: v for k, v in pair})
                return ret if len(ret) > 1 else ret[0]

            return _func_2

        return _deco


class BatchedTrain(BaseEstimator, metaclass=TimedMetaClass):
    def __init__(self, describe):
        """"""
        self.checker = None
        self._package_info = dict(
            python=py_ver,
            system=sys_ver(),
            numpy=np.__version__,
            torch=torch.__version__,
            xenonpy=__version__,
        )

    def __call__(self, *args, **kwargs):
        if not name:
            sig = md5(model.__str__().encode()).hexdigest()
            name = sig

    @property
    def package_info(self):
        """"""
        return deepcopy(self._package_info)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.stop()
        elapsed = str(timedelta(seconds=self.elapsed))
        self.logger('done runner <%s>: %s' % (self.__class__.__name__,
                                              datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        self.logger('total elapsed time: %s' % elapsed)
        logs = '\n'.join(self._logs)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        with open(str(Path(self.work_dir) / ('log_' + now + '.txt')), 'w') as f:
            f.write(logs)

    def __enter__(self):
        self.logger('start runner <%s> at %s' % (self.__class__.__name__,
                                                 datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        self._timer.start()
        return self


class RegressionRunner(BaseRunner, RegressorMixin):
    """
    Run model.
    """

    def __init__(self,
                 epochs=2000,
                 *,
                 cuda=False,
                 check_step=100,
                 log_step=0,
                 work_dir=None,
                 verbose=True,
                 describe=None):
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
        super(RegressionRunner, self).__init__(
            epochs, cuda=cuda, work_dir=work_dir, verbose=verbose, describe=describe)
        self._check_step = check_step
        self._log_step = log_step
        self._lr = 0.01
        self._lr_scheduler = None

        self.logger('Runner environment:', 'Running dir: {}'.format(self.work_dir),
                    'Epochs: {}'.format(self.epochs), 'Context: {}'.format(self._device),
                    'Check step: {}'.format(self._check_step), 'Log step: {}\n'.format(
                self._log_step))

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

    # @persist('y_true', 'y_pred')
    def post_predict(self, y_true, y_pred):
        y_pred = y_pred.cpu().detach().numpy()
        if y_true is not None:
            return y_true, y_pred
        return y_pred

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

            if self._log_step > 0 and t % self._log_step == 0 and i == 1:
                elapsed = str(timedelta(seconds=self.elapsed - start))
                start = self.elapsed
                self.logger('{}/{}, Loss={:.4f}, elapsed time: {}'.format(
                    t, self.epochs, loss, elapsed))
            if self._check_step > 0 and t % self._check_step == 0 and i == 1:
                self.checkpoint(mse_loss=loss.item())

        self.logger('Final loss={:.4f}'.format(loss), '')

        return y_train.cpu().detach().numpy(), y_pred.cpu().detach().numpy()
