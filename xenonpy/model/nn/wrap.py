#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from functools import partial

import torch as tc
from torch import nn

__all__ = ['Optim', 'LrScheduler', 'Init', 'L1']


class Optim(object):
    @staticmethod
    def sgd(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.SGD`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.SGD
        """
        return partial(tc.optim.SGD, *args, **kwargs)

    @staticmethod
    def ada_delta(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adadelta`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adadelta
        """
        return partial(tc.optim.Adadelta, *args, **kwargs)

    @staticmethod
    def ada_grad(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adagrad`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adagrad
        """
        return partial(tc.optim.Adagrad, *args, **kwargs)

    @staticmethod
    def adam(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adam`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adam
        """
        return partial(tc.optim.Adam, *args, **kwargs)

    @staticmethod
    def sparse_adam(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.SparseAdam`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.SparseAdam
        """
        return partial(tc.optim.SparseAdam, *args, **kwargs)

    @staticmethod
    def ada_max(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adamax`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adamax
        """
        return partial(tc.optim.Adamax, *args, **kwargs)

    @staticmethod
    def asgd(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.ASGD`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.ASGD
        """
        return partial(tc.optim.ASGD, *args, **kwargs)

    @staticmethod
    def lbfgs(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.LBFGS`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.LBFGS
        """
        return partial(tc.optim.LBFGS, *args, **kwargs)

    @staticmethod
    def rms_prop(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.RMSprop`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.RMSprop
        """
        return partial(tc.optim.RMSprop, *args, **kwargs)

    @staticmethod
    def r_prop(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Rprop`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Rprop
        """
        return partial(tc.optim.Rprop, *args, **kwargs)


class LrScheduler(object):
    @staticmethod
    def lambda_lr(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.LambdaLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.LambdaLR
        """
        return partial(tc.optim.lr_scheduler.LambdaLR, *args, **kwargs)

    @staticmethod
    def step_lr(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.StepLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.StepLR
        """
        return partial(tc.optim.lr_scheduler.StepLR, *args, **kwargs)

    @staticmethod
    def multi_step_lr(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.MultiStepLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.MultiStepLR
        """
        return partial(tc.optim.lr_scheduler.MultiStepLR, *args, **kwargs)

    @staticmethod
    def exponential_lr(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.ExponentialLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.ExponentialLR
        """
        return partial(tc.optim.lr_scheduler.ExponentialLR, *args, **kwargs)

    @staticmethod
    def reduce_lr_on_plateau(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        return partial(tc.optim.lr_scheduler.ReduceLROnPlateau, *args, **kwargs)


class Init(object):
    @staticmethod
    def uniform(*, scale=0.1):
        b = 1 * scale
        a = -b
        return partial(nn.init.uniform, a=a, b=b)


class L1(object):
    @staticmethod
    def conv(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.Conv1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.Conv1d
        """
        return partial(nn.Conv1d, *args, **kwargs)

    @staticmethod
    def linear(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.Linear`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.Linear
        """
        return partial(nn.Linear, *args, **kwargs)

    @staticmethod
    def batch_norm(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.BatchNorm1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.BatchNorm1d
        """
        return partial(nn.BatchNorm1d, *args, **kwargs)

    @staticmethod
    def instance_norm(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.InstanceNorm1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.InstanceNorm1d
        """
        return partial(nn.InstanceNorm1d, *args, **kwargs)
