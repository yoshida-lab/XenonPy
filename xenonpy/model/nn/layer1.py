# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from functools import partial

from torch import nn


class Wrap(object):
    @staticmethod
    def Conv1d(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.Conv1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.Conv1d
        """
        return partial(nn.Conv1d, *args, **kwargs)

    @staticmethod
    def Linear(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.Linear`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.Linear
        """
        return partial(nn.Linear, *args, **kwargs)

    @staticmethod
    def BatchNorm1d(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.BatchNorm1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.BatchNorm1d
        """
        return partial(nn.BatchNorm1d, *args, **kwargs)

    @staticmethod
    def InstanceNorm1d(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.InstanceNorm1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.InstanceNorm1d
        """
        return partial(nn.InstanceNorm1d, *args, **kwargs)


class Layer1d(nn.Module):
    """
    Base NN layer. This is a wrap around PyTorch.
    See here for details: http://pytorch.org/docs/master/nn.html#
    """

    def __init__(self, n_in: int, n_out: int, *,
                 p_drop=0.5,
                 layer_func=Wrap.Linear(bias=True),
                 act_func=nn.ReLU(),
                 batch_normalize=Wrap.BatchNorm1d(eps=1e-05, momentum=0.1, affine=True)
                 ):
        """
        Parameters
        ----------
        n_in: int
            Size of each input sample.
        n_out: int
            Size of each output sample
        p_drop: float
            Probability of an element to be zeroed. Default: 0.5
        layer_func: func
            Layers come with PyTorch.
        act_func: func
            Activation function.
        batch_normalize: func
            Normalization layers
        momentum
        """
        super().__init__()
        self.neuron = layer_func(n_in, n_out)
        self.batch_nor = None if not batch_normalize else batch_normalize(n_out)
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
