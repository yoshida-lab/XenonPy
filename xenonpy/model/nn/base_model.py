# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from pathlib import Path

import numpy as np
import torch as tc
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Variable as V

from .checkpoint import CheckPoint
from .layer import Layer1d


# from ...pipeline import combinator


class NNGenerator1d(object):
    def __init__(self, n_features: int, n_predict: int, *,
                 n_neuron: [int],
                 p_drop: [float] = (0.0,),
                 layer_func: [] = (nn.Linear,),
                 act_func: [] = (nn.ReLU(),),
                 batch_normalize: [bool] = (False,),
                 momentum: [float] = (0.1,)
                 ):
        self.n_in, self.n_out = n_features, n_predict

        # save parameters
        self.n_neuron = n_neuron
        self.p_drop = p_drop
        self.layer_func = layer_func
        self.act_func = act_func
        self.batch_normalize = batch_normalize
        self.momentum = momentum

        # prepare layer container
        # calculate layer's variety
        self.layers = list()
        self.layers_len = 0
        self.__lay_vars()

    def __lay_vars(self):
        for n in self.n_neuron:
            for p in self.p_drop:
                for l in self.layer_func:
                    for a in self.act_func:
                        for b in self.batch_normalize:
                            for m in self.momentum:
                                layer = dict(n_in=0,
                                             n_out=n,
                                             p_drop=p,
                                             layer_func=l,
                                             act_func=a,
                                             batch_normalize=b,
                                             momentum=m)
                                self.layers.append(layer)
        self.layers_len = len(self.layers)

    def __call__(self, hidden=3):
        ids = np.random.randint(0, self.layers_len, hidden)
        layers = list()

        # set layers
        self.layers[ids[0]]['n_in'] = self.n_in
        n_in = self.n_in
        for i in ids:
            self.layers[i]['n_in'] = n_in
            layers.append(Layer1d(**self.layers[i]))
            n_in = self.layers[i]['n_out']
        out_layer = Layer1d(n_in=n_in, n_out=self.n_out, act_func=None)
        layers.append(out_layer)

        return nn.Sequential(*layers)


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
        if not isinstance(x, V):
            x = V(x, requires_grad=False)
        _, col = x.size()
        if not isinstance(y, V):
            y = V(y, requires_grad=False)

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
        if not isinstance(x, V):
            x = V(x)
        pre_y = self.model(x)
        if y:
            if not isinstance(y, V):
                y = V(y)
            print(self.loss_func(pre_y, y))
        return pre_y
