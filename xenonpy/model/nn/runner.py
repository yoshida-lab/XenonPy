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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from torch.autograd import Variable as Var

from .checker import Checker
from .wrap import Init
from ...utils import Stopwatch


class ModelRunner(BaseEstimator, RegressorMixin):
    """
    Run model.
    """

    def __init__(self, epochs=2000, *,
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
        self._model = None
        self._model_name = None
        self._loss_func = None
        self._optim = None
        self._lr = None
        self._lr_scheduler = None

    def __enter__(self):
        if self._verbose:
            print('Runner environment:')
            print('Running dir: {}'.format(self._work_dir))
            print('Epochs: {}'.format(self._epochs))
            print('Context: {}'.format(self._ctx.upper()))
            print('Check step: {}'.format(self._check_step))
            print('Log step: {}\n'.format(self._log_step))
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
                print('init weight -> {}'.format(m))
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
            model.apply(_init_weight)
        self._checker = Checker(name, self._work_dir)
        self._model = model
        self._model_name = name
        self._loss_func = loss_func
        self._optim = optim
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._checker.init_model = model

    @staticmethod
    def _d2tv(data):
        if isinstance(data, df):
            data = tc.from_numpy(data.as_matrix()).type(tc.FloatTensor)
        elif isinstance(data, np.ndarray):
            data = tc.from_numpy(data).type(tc.FloatTensor)
        else:
            raise TypeError('need to be <numpy.ndarray> or <pandas.DataFrame> but got {}'.format(type(data)))
        return Var(data, requires_grad=False)

    def _train(self, x_train, y_train):
        stopwatch = Stopwatch()
        # optimization
        optim = self._optim(self._model.parameters(), lr=self._lr)

        # adjust learning rate
        scheduler = self._lr_scheduler(optim) if self._lr_scheduler else None

        # start
        loss, y_pred = None, None
        print('=======start training=======')
        print('Model name: {}\n'.format(self._model_name))

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
                self._checker(model_state=self._model.state_dict(),
                              epochs=t,
                              loss=loss.data[0],
                              elapsed=stopwatch.count)

        print('\n=======over training=======')
        print('Final loss={:.4f}\n'.format(loss.data[0]))

        # save last results
        self._checker(model_state=self._model.state_dict(),
                      epochs=self._epochs,
                      loss=loss.data[0],
                      elapsed=stopwatch.count)

    def fit(self, x_train, y_train, x_id=None, y_id=None):
        """
        Fit Neural Network model

        Parameters
        ----------
        x_train: ``numpy.ndarray`` or ``pandas.DataFrame``
            Training data.
        y_train: ``numpy.ndarray`` or ``pandas.DataFrame``
            Target values.
        x_id: list-like
            Row id for train
        y_id: list-like
            Row id for test

        Returns
        -------
        self:
            returns an instance of self.
        """
        self._checker.train_data(x_train=x_train, y_train=y_train, x_id=x_id, y_id=y_id)

        # transform to torch tensor
        x_train = self._d2tv(x_train)
        y_train = self._d2tv(y_train)

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
        self._train(x_train, y_train)

        # move back to cpu
        self._model.cpu()
        self._checker.trained_model = self._model
        self._checker.describe = dict(self._add_info,
                                      structure=str(self._model),
                                      running_at=self._work_dir,
                                      created_at=dt.now().strftime('-%Y-%m-%d_%H-%M-%S_%f'))

        return self

    def predict(self, x_test, y_test):
        # prepare x
        x_test = self._d2tv(x_test)

        # if use CUDA acc
        if self._ctx.lower() == 'gpu':
            if tc.cuda.is_available():
                self._model.cuda()
                x_test = x_test.cuda()
            else:
                warn('No cuda environment, use cpu fallback.', RuntimeWarning)
        else:
            self._model.cpu()
        # prediction
        y_true, y_pred = y_test.ravel(), self._model(x_test).cpu().data.numpy().ravel()
        # move back to cpu
        self._model.cpu()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        pr, p_val = pearsonr(y_true, y_pred)
        metrics = dict(
            mae=mae,
            rmse=rmse,
            r2=r2,
            pearsonr=pr,
            p_value=p_val
        )

        self._checker.add_predict(
            x_test=x_test.cpu().data.numpy(),
            y_test=y_test,
            y_pred=y_pred,
            metrics=metrics
        )

        return y_true, y_pred, metrics

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
        val = dict(model=self._model, **kwargs)
        joblib.dump(val, fpath)
