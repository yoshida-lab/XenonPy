# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
from datetime import datetime as dt
from warnings import warn

import numpy as np
import torch as tc
import torch.nn as nn
from pandas import DataFrame as df
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.autograd import Variable as Var

from .checker import Checker
from .wrap import Init
from ...preprocess.data_select import DataSplitter
from ...preprocess.transform import Scaler
from ...utils.functional import Stopwatch


class ModelRunner(BaseEstimator):
    """
    Run model.
    """

    def __init__(self, epochs=2000, *,
                 xy_scaler=None,
                 ctx='cpu',
                 check_step=100,
                 log_step=0,
                 work_dir=None,
                 verbose=True,
                 additional_info=None
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
        self._add_info = additional_info if additional_info else {}
        self._verbose = verbose
        self._check_step = check_step
        self._ctx = ctx
        self._epochs = epochs
        self._log_step = log_step
        self._work_dir = work_dir
        self._checker = None
        self._splitter = None
        self._model = None
        self._model_name = None
        self._loss_func = None
        self._optim = None
        self._lr = None
        self._lr_scheduler = None
        self._x_torch_type = self._y_torch_type = None
        if xy_scaler:
            self._x_scaler, self._y_scaler = self._check_xy_scaler(xy_scaler)
        else:
            self._x_scaler, self._y_scaler = None, None

        if self._verbose:
            print('Runner environment:')
            print('Running dir: {}'.format(self._work_dir))
            print('Epochs: {}'.format(self._epochs))
            print('Context: {}'.format(self._ctx.upper()))
            print('Check step: {}'.format(self._check_step))
            print('Log step: {}\n'.format(self._log_step))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self

    def __call__(self, model, name=None, *,
                 init_weight=Init.uniform(scale=0.1),
                 loss_func=None,
                 optim=None,
                 lr=0.001,
                 lr_scheduler=None):

        def _init_weight(m):
            if isinstance(m, nn.Linear):  # fixme: not only linear
                init_weight(m.weight)

        # model must inherit form nn.Module
        if not isinstance(model, nn.Module):
            raise ValueError(
                'Runner need a `torch.nn.Module` instance as first parameter but got {}'.format(type(model)))

        if not name:
            from hashlib import md5
            sig = md5(model.__str__().encode()).hexdigest()
            name = sig
        if init_weight:
            if self._verbose:
                print('init weight with {}'.format(init_weight))
            model.apply(_init_weight)
        self._checker = Checker(name, self._work_dir)
        self._model = model
        self._model_name = name
        self._loss_func = loss_func
        self._optim = optim
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._checker.init_model = model
        self._checker.save(runner=dict(
            add_info=self._add_info,
            verbose=self._verbose,
            check_step=self._check_step,
            ctx=self._ctx,
            epochs=self._epochs,
            log_step=self._log_step,
            work_dir=self._work_dir,
            model_name=self._model_name,
            loss_func=self._loss_func,
            optim=self._optim,
            lr=self._lr,
            lr_scheduler=self._lr_scheduler
        ))

    @staticmethod
    def _check_xy_scaler(scaler):
        if isinstance(scaler, (tuple, list)):
            if not (isinstance(scaler[0], Scaler) and isinstance(scaler[1], Scaler)):
                raise TypeError('must be a Scaler but got ({}, {})'.format(type(scaler[0]), type(scaler[1])))
            if scaler[0].value.shape[0] != scaler[1].value.shape[0]:
                raise ValueError('X y should have same row size (shape[0])')
            return scaler[0], scaler[1]
        else:
            raise ValueError('parameter scaler must be a tuple or list with size 2 but got {}'.format(scaler))

    @property
    def xy_scaler(self):
        return self._x_scaler, self._y_scaler

    @xy_scaler.setter
    def xy_scaler(self, scaler):
        self._x_scaler, self._y_scaler = self._check_xy_scaler(scaler)

    @staticmethod
    def _d2tv(data, torch_type):
        if isinstance(data, df):
            if torch_type is None:
                data = tc.from_numpy(data.as_matrix()).type(tc.FloatTensor)
            else:
                data = tc.from_numpy(data.as_matrix()).type(torch_type)
        elif isinstance(data, np.ndarray):
            if torch_type is None:
                data = tc.from_numpy(data).type(tc.FloatTensor)
            else:
                data = tc.from_numpy(data).type(torch_type)
        else:
            raise TypeError('need to be <numpy.ndarray> or <pandas.DataFrame> but got {}'.format(type(data)))
        return Var(data, requires_grad=False)

    @staticmethod
    def metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        pr, p_val = pearsonr(y_true, y_pred)
        return dict(
            mae=mae,
            rmse=rmse,
            r2=r2,
            pearsonr=pr,
            p_value=p_val
        )

    def _train(self, x_train, y_train):
        stopwatch = Stopwatch()
        # optimization
        optim = self._optim(self._model.parameters(), lr=self._lr)

        # adjust learning rate
        scheduler = self._lr_scheduler(optim) if self._lr_scheduler else None

        # start
        loss, y_pred = None, None
        print('Model name: {}\n'.format(self._model_name))
        print('=======start training=======')

        for t in range(self._epochs):
            if scheduler and not isinstance(scheduler, tc.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
            y_pred = self._model(x_train)
            loss = self._loss_func(y_pred, y_train)
            if scheduler and isinstance(scheduler, tc.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self._log_step > 0 and t % self._log_step == 0:
                print('at step[{}/{}], Loss={:.4f}, elapsed time: {}'.format(t, self._epochs,
                                                                             loss.data[0],
                                                                             stopwatch.count))
            if self._check_step > 0 and t % self._check_step == 0:
                self._checker(model_state=self._model.state_dict(), epochs=t)

        print('\n=======over training=======')

        # save last results
        self._checker(model_state=self._model.state_dict(), epochs=self._epochs)
        y_train, y_pred = y_train.cpu().data.numpy(), y_pred.cpu().data.numpy()
        metrics = self.metrics(y_train, y_pred)

        if self._verbose:
            print('Final loss={:.4f}\n'.format(loss.data[0]))
            print('Mean absolute error with train data: %s' % metrics['mae'])
            print('Root mean squared error with train data: %s' % metrics['rmse'])

        return y_train, y_pred, metrics

    def fit(self, *xy, test_size=0.2, x_torch_type=None, y_torch_type=None):
        """
        Fit Neural Network model

        Parameters
        ----------
        xy: None or list or ``numpy.ndarray`` or ``pandas.DataFrame``.
            Training data.
            If None, will split data in ``self.xy_scaler`` with ``test_size=0.2``.
            If ``self.xy_scaler``  is None, raise ``ValueError``.
            If list, use as indices to sample training data from `self.xy_scaler``.
            If ``self.xy_scaler``  is None, raise ``ValueError``.
            Also can input x y training data directly in ``numpy.ndarray`` or ``pandas.DataFrame``.
        test_size: float, int, None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. By default, the value is set to 0.2.
        x_torch_type: torch data type
            Default is torch.FloatTensor.
        y_torch_type: torch data type
            Default is torch.FloatTensor.
        Returns
        -------
        self:
            returns an instance of self.
        """
        if len(xy) == 0:
            if self._x_scaler and self._y_scaler:
                ds = DataSplitter(self._x_scaler.np_value, test_size=test_size)
                x_train, y_train = ds.split_data(self._x_scaler.np_value, self._y_scaler.np_value, test=False)
                self._checker.save_data(x_scaler=self._x_scaler, y_scaler=self._y_scaler, splitter=ds)
                self._splitter = ds
            else:
                raise ValueError('No data for training, please set ``xy_scaler`` or input training data directly')
        else:
            if len(xy) != 2:
                raise ValueError('xy size must be 2 but got {}'.format(len(xy)))
            x_train, y_train = xy
            self._checker.save_data(x_train=x_train, y_train=y_train)

        # transform to torch tensor
        x_train = self._d2tv(x_train, x_torch_type)
        y_train = self._d2tv(y_train, y_torch_type)

        # if use CUDA acc
        if self._ctx.lower() == 'gpu':
            if tc.cuda.is_available():
                self._model.cuda()
                x_train = x_train.cuda()
                y_train = y_train.cuda()
            else:
                warn('No cuda environment, use cpu fallback.', RuntimeWarning)
        else:
            self._model.cpu()

        # train
        result = self._train(x_train, y_train)

        # move back to cpu
        self._model.cpu()
        self._checker.trained_model = self._model
        self._checker.describe = dict(self._add_info,
                                      structure=str(self._model),
                                      running_at=self._work_dir,
                                      created_at=dt.now().strftime('-%Y-%m-%d_%H-%M-%S_%f'))

        return result

    def predict(self, *xy, x_torch_type=None):
        save_data = False
        if len(xy) == 0:
            if self._x_scaler and self._y_scaler:
                x_test, y_test = self._splitter.split_data(self._x_scaler.np_value, self._y_scaler.np_value,
                                                           train=False)
            else:
                raise ValueError('No data for test, please set ``xy_scaler`` or input test data directly')
        else:
            if len(xy) != 2:
                raise ValueError('xy size must be 2 but got {}'.format(len(xy)))
            x_test, y_test = xy
            save_data = True
        # prepare x
        x_test_ = self._d2tv(x_test, x_torch_type)

        # if use CUDA acc
        if self._ctx.lower() == 'gpu':
            if tc.cuda.is_available():
                self._model.cuda()
                x_test_ = x_test_.cuda()
            else:
                warn('No cuda environment, use cpu fallback.', RuntimeWarning)
        else:
            self._model.cpu()
        # prediction
        y_pred = self._model(x_test_).cpu().data.numpy()
        metrics = self.metrics(y_test, y_pred)

        if save_data:
            self._checker.add_predict(
                x_test=x_test,
                y_test=y_test,
                y_pred=y_pred,
                metrics=metrics
            )
        else:
            self._checker.add_predict(
                y_pred=y_pred,
                metrics=metrics
            )

        return y_test, y_pred, metrics

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
