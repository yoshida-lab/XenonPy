#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import math
from typing import Union, Tuple, Callable, Any, Optional

from torch import nn

__all__ = ['LinearLayer', 'SequentialLinear']


class LinearLayer(nn.Module):
    """
    Base NN layer. This is a wrap around PyTorch.
    See here for details: http://pytorch.org/docs/master/nn.html#
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *,
                 dropout: float = 0.,
                 activation_func: Callable = nn.ReLU(),
                 normalizer: Union[float, None] = .1
                 ):
        """
        Parameters
        ----------
        in_features:
            Size of each input sample.
        out_features:
            Size of each output sample
        dropout: float
            Probability of an element to be zeroed. Default: 0.5
        activation_func: func
            Activation function.
        normalizer: func
            Normalization layers
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.dropout = nn.Dropout(dropout)
        self.normalizer = None if not normalizer else nn.BatchNorm1d(out_features, normalizer)
        self.activation = None if not activation_func else activation_func

    def forward(self, x):
        _out = self.linear(x)
        if self.dropout:
            _out = self.dropout(_out)
        if self.normalizer:
            _out = self.normalizer(_out)
        if self.activation:
            _out = self.activation(_out)
        return _out


class SequentialLinear(nn.Module):
    """
    Sequential model with linear layers and configurable other hype-parameters.
    e.g. ``dropout``, ``hidden layers``

    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *,
                 h_neurons: Union[Tuple[float, ...], Tuple[int, ...]] = (),
                 h_bias: Union[bool, Tuple[bool, ...]] = True,
                 h_dropouts: Union[float, Tuple[float, ...]] = 0.1,
                 h_normalizers: Union[float, None, Tuple[Optional[float], ...]] = 0.1,
                 h_activation_funcs: Union[Callable, None, Tuple[Optional[Callable], ...]] = nn.ReLU(),
                 ):
        """

        Parameters
        ----------
        in_features
            Size of input.
        out_features
            Size of output.
        bias
            Enable ``bias`` in input layer.
        h_neurons
            Number of neurons in hidden layers.
            Can be a tuple of floats. In that case,
            all these numbers will be used to calculate the neuron numbers.
            e.g. (0.5, 0.4, ...) will be expanded as (in_features * 0.5, in_features * 0.4, ...)
        h_bias
            ``bias`` in hidden layers.
        h_dropouts
            Probabilities of dropout in hidden layers.
        h_normalizers
            Momentum of batched normalizers in hidden layers.
        h_activation_funcs
            Activation functions in hidden layers.
        """
        super().__init__()
        self._h_layers = len(h_neurons)
        if self._h_layers > 0:
            if isinstance(h_neurons[0], float):
                tmp = [in_features]
                for i, ratio in enumerate(h_neurons):
                    num = math.ceil(in_features * ratio)
                    tmp.append(num)
                neurons = tuple(tmp)

            elif isinstance(h_neurons[0], int):
                neurons = (in_features,) + tuple(h_neurons)
            else:
                raise RuntimeError('illegal parameter type of <h_neurons>')

            activation_funcs = self._check_input(h_activation_funcs)
            normalizers = self._check_input(h_normalizers)
            dropouts = self._check_input(h_dropouts)
            bias = (bias,) + self._check_input(h_bias)

            for i in range(self._h_layers):
                setattr(self, f'layer_{i}', LinearLayer(
                    in_features=neurons[i],
                    out_features=neurons[i + 1],
                    bias=bias[i],
                    dropout=dropouts[i],
                    activation_func=activation_funcs[i],
                    normalizer=normalizers[i]
                ))

            self.output = nn.Linear(neurons[-1], out_features, bias[-1])
        else:
            self.output = nn.Linear(in_features, out_features, bias)

    def _check_input(self, i):
        if isinstance(i, Tuple):
            if len(i) != self._h_layers:
                raise RuntimeError(
                    f'number of parameter not consistent with number of layers, '
                    f'input is {len(i)} but need to be {self._h_layers}')
            return tuple(i)
        else:
            return tuple([i] * self._h_layers)

    def forward(self, x: Any) -> Any:
        for i in range(self._h_layers):
            x = getattr(self, f'layer_{i}')(x)
        return self.output(x)
