# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from pathlib import Path

import torch as tc
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Variable as Var

from .checkpoint import CheckPoint


# from ...pipeline import combinator
class Layer1d(nn.Module):
    def __init__(self, n_in: int, n_out: int, *,
                 p_drop: float = 0.0,
                 layer_func=nn.Linear,
                 act_func=nn.ReLU(),
                 batch_normalize: bool = False,
                 momentum: float = 0.1,
                 lr: float = None
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
    def __init__(self, model: nn.Module, *,
                 lr: float = 0.01,
                 loss_func=None,
                 optim=None,
                 epochs: int = 2000,
                 verbose: int = 0,
                 ctx: str = 'cpu',
                 checkstep: int = 100,
                 check: tuple = ('epochs', 'loss'),
                 save_to: str = ''
                 ):
        parent = Path(save_to)
        if not parent.exists():
            parent.mkdir()
        self.save_snap = str(Path(save_to) / 'snap.pkl')
        self.save_model = str(Path(save_to) / 'model.pkl')
        self.checkstep = checkstep
        self.check = check
        self.trained = False
        self.model = model
        self.checkpoint = CheckPoint(model, *check)
        self.x = None
        self.y = None
        self.lr = lr
        self.ctx = ctx
        self.loss_func = loss_func
        self.optim = optim
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, x, y=None):

        # transform to torch tensor
        if not isinstance(x, Var):
            x = Var(x, requires_grad=False)
        _, col = x.size()
        if not isinstance(y, Var):
            y = Var(y, requires_grad=False)

        # optimization
        optim = self.optim(self.model.parameters(), lr=self.lr)

        # if use CUDA acc
        if self.ctx == 'GPU' and tc.cuda.is_available():
            self.model.cuda()
            x = x.cuda()
            y = y.cuda()

        # train
        loss = None
        for t in range(self.epochs):
            pre_y = self.model(x)
            loss = self.loss_func(pre_y, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if self.verbose and t % self.verbose == 0:
                print('at step: {}, Loss={:.4f}'.format(t, loss.data[0]))
            if self.checkstep > 0 and t % self.checkstep == 0:
                self.checkpoint(epochs=t, loss=loss.data[0])

        self.trained = True
        if self.checkstep > 0:
            self.checkpoint.save(self.save_snap, self.save_model)
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
