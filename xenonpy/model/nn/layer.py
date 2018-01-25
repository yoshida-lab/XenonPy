# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import torch.nn as nn


class Layer1d(nn.Module):
    def __init__(self, *,
                 n_in: int,
                 n_out: int,
                 p_drop: float = 0.0,
                 layer_func=nn.Linear,
                 act_func=nn.ReLU(),
                 batch_normalize=False,
                 momentum=0.1
                 ):
        super().__init__()
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

