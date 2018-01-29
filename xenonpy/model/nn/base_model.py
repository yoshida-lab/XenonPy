# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from itertools import product

import numpy as np
import torch as tc
import torch.nn as nn
from collections import namedtuple
from pathlib import Path
from sklearn.base import BaseEstimator, RegressorMixin
from torch.autograd import Variable as V

from .checkpoint import CheckPoint
from .layer import Layer1d


# from ...pipeline import combinator


class NNGenerator1d(object):
    """
    Generate random model from the supplied parameters.
    """
    def __init__(self, n_features: int, n_predict: int, *,
                 n_neuron: [int],
                 p_drop: [float] = (0.0,),
                 layer_func: [] = (nn.Linear,),
                 act_func: [] = (nn.ReLU(),),
                 lr: [float] = (None,),
                 batch_normalize: [bool] = (False,),
                 momentum: [float] = (0.1,)
                 ):
        """

        Parameters
        ----------
        n_features: int
            Input dimension.
        n_predict: int
            Output dimension.
        n_neuron: int-list
            Number of neuron.
        p_drop: float-list
            Dropout rate.
        layer_func: func-list
            Layer functions. such like: :class:`torch.nn.Linear`.
        act_func: func-list
            Activation functions. such like: :class:``torch.nn.ReLU`.
        lr: float-list
            Learning rates.
        batch_normalize: bool-list
            Batch Normalization. such like: :class:`torch.nn.BatchNorm1d`.
        momentum: float-list
            The value used for the running_mean and running_var computation.
        """
        self.n_in, self.n_out = n_features, n_predict

        # save parameters
        self.lr = lr
        self.n_neuron = n_neuron
        self.p_drop = p_drop
        self.layer_func = layer_func
        self.act_func = act_func
        self.batch_normalize = batch_normalize
        self.momentum = momentum

        # prepare layer container
        # calculate layer's variety
        self.layer_var = list(product(n_neuron, p_drop, layer_func, act_func, batch_normalize, momentum, lr))

    def __call__(self, hidden: int = 3, n_sample: int = 0, scheduler=None):
        """
        Generate sample model.

        Parameters
        ----------
        hidden: int
            Number of hidden layers.
        n_sample: int
            Number of model sample
        scheduler:
            A function be used to determining the layer properties from previous layer.

                >>> # index: layer index in a model; pars: parameters of previous layer as dict.
                >>> # include: n_neuron, p_drop, layer_func, act_func, lr, batch_normalize, momentum
                >>> scheduler = lambda index, pars: pars

        Returns
        -------
        ret: iterable
            Samples as generator
        """
        layer_paras = namedtuple('LayerParas',
                                 ['n_in',
                                  'n_out',
                                  'p_drop',
                                  'layer_func',
                                  'act_func',
                                  'batch_normalize',
                                  'momentum',
                                  'lr']
                                 )
        layers_len = len(self.layer_var)
        ids = np.random.randint(0, layers_len, hidden)
        layers = list()

        # set layers
        n_in = self.n_in
        for i in ids:
            layer = (n_in,) + self.layer_var[i]
            layers.append(Layer1d(**(layer_paras(*layer)._asdict())))
            n_in = self.layer_var[i][0]
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
