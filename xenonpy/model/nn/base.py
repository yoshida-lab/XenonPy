#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import math
from datetime import datetime, timedelta
from pathlib import Path
from platform import version as sys_ver
from sys import version as py_ver

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from sklearn.base import BaseEstimator

from .checker import Checker
from ..._conf import __version__
from ...utils import TimedMetaClass


class BaseRunner(BaseEstimator, metaclass=TimedMetaClass):

    def __init__(self,
                 epochs=2000,
                 *,
                 cuda: bool or int or str = False,
                 work_dir='.',
                 verbose=True,
                 describe=None):
        self._epochs = epochs
        if isinstance(cuda, bool):
            self._device = torch.device('cuda') if cuda else torch.device('cpu')
        elif isinstance(cuda, int):
            self._device = torch.device('cuda', cuda)
        else:
            self._device = torch.device(cuda)
        if not Path(work_dir).exists():
            Path(work_dir).mkdir()
        self._verbose = verbose
        self._work_dir = work_dir
        self._model = None
        self._model_name = None
        self._checker = None
        self._logs = []
        self._describe = describe or dict(
            python=py_ver,
            system=sys_ver(),
            numpy=np.__version__,
            torch=torch.__version__,
            xenonpy=__version__,
            workspace=work_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timer.stop()
        elapsed = str(timedelta(seconds=self.elapsed))
        self.logger('done runner <%s>: %s' % (self.__class__.__name__,
                                              datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
        self.logger('total elapsed time: %s' % elapsed)
        logs = '\n'.join(self._logs)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        with open(str(Path(self._work_dir) / ('log_' + now + '.txt')), 'w') as f:
            f.write(logs)

    def __enter__(self):
        self.logger('start runner <%s> at %s' % (self.__class__.__name__,
                                                 datetime.now().strftime('%Y/%m/%d %H:%M:%S')))
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

    def __call__(self, model, name=None, **kwargs):
        """"""
        # model must inherit form nn.Module
        if not isinstance(model, torch.nn.Module):
            raise TypeError('Need <torch.nn.Module> instance but got %s' % type(model))

        if not name:
            from hashlib import md5
            sig = md5(model.__str__().encode()).hexdigest()
            name = sig
        self._checker = Checker(name, self._work_dir)
        self._model = model
        self._model_name = name
        self.describe(model_struct=str(model), model_name=name)
        self._checker.init_model = model
        self._checker.save(
            runner=dict(epochs=self._epochs, verbose=self._verbose, workspace=self._work_dir))

        if kwargs:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def describe(self, **info):
        """
        Add some additional description to runner.
        Description will be saved automaticly.

        Parameters
        ----------
        info: dict
            Additional information to describe this model training.
        """
        self._describe = dict(self._describe, **info)

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
            if torch_type is None:
                torch_type = torch.float
            if isinstance(data, pd.DataFrame):
                return torch.from_numpy(data.as_matrix()).to(torch_type)
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).to(torch_type)
            else:
                raise TypeError(
                    'Need <numpy.ndarray> or <pandas.DataFrame> but got %s' % type(data))

        return tuple([_tensor(data_, type_) for data_, type_ in data_and_type])

    def to_device(self, *tensor):
        # if use CUDA acc
        if self._device.type != 'cpu':
            return tuple([t.cuda(self._device, True) for t in tensor])
        return tuple([t.cpu() for t in tensor])

    def batch_tensor(self, *data, batch_size=0.2, shuffle=True, num_worker=0, pin_memory=True):
        # batch_size
        if not data:
            return None
        if not all([isinstance(o, (pd.DataFrame, np.ndarray)) for o, _ in data]):
            raise TypeError('Need <numpy.ndarray> or <pandas.DataFrame>')
        if isinstance(batch_size, float):
            batch_size = math.ceil(data[0][0].shape[0] * batch_size)

        return Data.DataLoader(
            dataset=Data.TensorDataset(*self.tensor(*data)),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_worker,
            pin_memory=pin_memory)

    def logger(self, *info):
        log = '|> ' + '\n|> '.join(info)
        self._logs.append(log)
        if self._verbose:
            print(log)

    def fit(self,
            x_train=None,
            y_train=None,
            *,
            data_loader=None,
            x_dtype=torch.float,
            y_dtype=torch.float):
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
                    yield y_, self._model(x_), t, 1
                return

            if data_loader:
                for t in range(self._epochs):
                    for i, (x_, y_) in enumerate(data_loader):
                        x_, y_ = self.to_device(x_, y_)
                        yield y_, self._model(x_), t, i
                return

        self.describe(start=datetime.now().strftime('%Y/%m/%d %H:%M:%S'))

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

        self.describe(done=now)
        self._checker.save(describe=self._describe)
        self._checker.trained_model = self._model

        return ret

    def predict(self, x_test, y_test=None, *, x_dtype=torch.float, y_dtype=torch.float):
        """

        Parameters
        ----------
        x_test: DataFrame, ndarray
            Input data for test..
        y_test: DataFrame, ndarray, optional
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
        x_test, = self.to_device(*self.tensor((x_test, x_dtype)))

        # prediction
        self._model.to(self._device)
        y_true, y_pred = y_test, self._model(x_test)
        return self.post_predict(y_true, y_pred)

    @classmethod
    def from_checker(cls, checker, checkpoint=None):
        runner = checker.last('runner')
        ret = cls(
            runner['epochs'],
            work_dir=runner['workspace'],
            verbose=runner['verbose'],
            describe=checker.describe)
        ret._checker = checker
        ret._model_name = checker.model_name
        if not checkpoint:
            ret._model = checker.trained_model if checker.trained_model else checker.init_model
        else:
            model_state, _ = checker[checkpoint]
            ret._model = checker.init_model.load_state_dict(model_state)
        return ret
