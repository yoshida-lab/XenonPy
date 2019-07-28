#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from torch import nn

from .wrap import L1

__all__ = ['Layer1d']


class Layer1d(nn.Module):
    """
    Base NN layer. This is a wrap around PyTorch.
    See here for details: http://pytorch.org/docs/master/nn.html#
    """

    def __init__(self, n_in, n_out, *,
                 drop_out=0.,
                 layer_func=L1.linear(bias=True),
                 act_func=nn.ReLU(),
                 batch_nor=L1.batch_norm(eps=1e-05, momentum=0.1, affine=True)
                 ):
        """
        Parameters
        ----------
        n_in: int
            Size of each input sample.
        n_out: int
            Size of each output sample
        drop_out: float
            Probability of an element to be zeroed. Default: 0.5
        layer_func: func
            Layers come with PyTorch.
        act_func: func
            Activation function.
        batch_nor: func
            Normalization layers
        """
        super().__init__()
        self.layer = layer_func(n_in, n_out)
        self.batch_nor = None if not batch_nor else batch_nor(n_out)
        self.act_func = None if not act_func else act_func
        self.dropout = None if drop_out == 0. else nn.Dropout(drop_out)

    def forward(self, *x):
        _out = self.layer(*x)
        if self.dropout:
            _out = self.dropout(_out)
        if self.batch_nor:
            _out = self.batch_nor(_out)
        if self.act_func:
            _out = self.act_func(_out)
        return _out
