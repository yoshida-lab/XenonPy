# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from itertools import product
from torch import nn

import numpy as np
from collections import namedtuple
from random import sample

from . import Layer1d


class Generator1d(object):
    """
    Generate random model from the supplied parameters.
    """

    def __init__(self, n_features: int, n_predict: int, *,
                 n_neuron,
                 p_drop=(0.0,),
                 layer_func=(nn.Linear,),
                 act_func=(nn.ReLU(),),
                 lr=(None,),
                 batch_normalize=(False,),
                 momentum=(0.1,)
                 ):
        """

        Parameters
        ----------
        n_features: int
            Input dimension.
        n_predict: int
            Output dimension.
        n_neuron: [int]
            Number of neuron.
        p_drop: [float]
            Dropout rate.
        layer_func: [func]
            Layer functions. such like: :class:`torch.nn.Linear`.
        act_func: [func]
            Activation functions. such like: :class:`torch.nn.ReLU`.
        lr: [float]
            Learning rates.
        batch_normalize: [bool]
            Batch Normalization. such like: :class:`torch.nn.BatchNorm1d`.
        momentum: [float]
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

    def __call__(self, hidden: int, *, n_models: int = 0, scheduler=None):
        """
        Generate sample model.

        Parameters
        ----------
        hidden: int
            Number of hidden layers.
        n_models: int
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
        named_paras = ['n_in', 'n_out', 'p_drop', 'layer_func', 'act_func', 'batch_normalize', 'momentum', 'lr']
        layer = namedtuple('LayerParas', named_paras)
        layer_len = len(self.layer_var)

        if scheduler is None:
            indices = list(product(np.arange(layer_len), repeat=hidden))
            indices_len = layer_len ** hidden

            if n_models == 0 or n_models >= indices_len:
                n_models = indices_len

            # sampling indices
            samples = sample(indices, n_models)
            while True:
                try:
                    ids = samples.pop()
                except IndexError:
                    raise StopIteration()

                # set layers
                layers = list()
                n_in = self.n_in
                for i in ids:
                    layer_ = layer(*((n_in,) + self.layer_var[i]))._asdict()
                    layers.append(Layer1d(**layer_))
                    n_in = self.layer_var[i][0]
                out_layer = Layer1d(n_in=n_in, n_out=self.n_out, act_func=None)
                layers.append(out_layer)

                yield nn.Sequential(*layers)

        else:
            if n_models == 0 or n_models >= layer_len:
                n_models = layer_len

            samples = sample(np.arange(layer_len).tolist(), n_models)

            while True:
                try:
                    i = samples.pop()
                except IndexError:
                    raise StopIteration()

                layers = list()
                n_in = self.n_in
                paras = self.layer_var[i]
                layer_ = layer(*((n_in,) + paras))._asdict()
                layers.append(Layer1d(**layer_))
                n_in = layer_['n_out']
                for n in np.arange(hidden - 1):
                    layer_ = scheduler(n + 1, layer_)
                    layer_['n_in'] = n_in
                    layers.append(Layer1d(**layer_))
                    n_in = layer_['n_out']
                out_layer = Layer1d(n_in=n_in, n_out=self.n_out, act_func=None)
                layers.append(out_layer)

                yield nn.Sequential(*layers)
