# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from functools import partial

import torch as tc


class SL(object):
    def __init__(self):
        self.load = tc.load
        self.dump = tc.save


class Optim(object):
    @staticmethod
    def SGD(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.SGD`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.SGD
        """
        return partial(tc.optim.SGD, *args, **kwargs)

    @staticmethod
    def Adadelta(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adadelta`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adadelta
        """
        return partial(tc.optim.Adadelta, *args, **kwargs)

    @staticmethod
    def Adagrad(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adagrad`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adagrad
        """
        return partial(tc.optim.Adagrad, *args, **kwargs)

    @staticmethod
    def Adam(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adam`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adam
        """
        return partial(tc.optim.Adam, *args, **kwargs)

    @staticmethod
    def SparseAdam(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.SparseAdam`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.SparseAdam
        """
        return partial(tc.optim.SparseAdam, *args, **kwargs)

    @staticmethod
    def Adamax(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Adamax`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Adamax
        """
        return partial(tc.optim.Adamax, *args, **kwargs)

    @staticmethod
    def ASGD(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.ASGD`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.ASGD
        """
        return partial(tc.optim.ASGD, *args, **kwargs)

    @staticmethod
    def LBFGS(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.LBFGS`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.LBFGS
        """
        return partial(tc.optim.LBFGS, *args, **kwargs)

    @staticmethod
    def RMSprop(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.RMSprop`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.RMSprop
        """
        return partial(tc.optim.RMSprop, *args, **kwargs)

    @staticmethod
    def Rprop(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.Rprop`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.Rprop
        """
        return partial(tc.optim.Rprop, *args, **kwargs)


class LrScheduler(object):
    @staticmethod
    def LambdaLR(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.LambdaLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.LambdaLR
        """
        return partial(tc.optim.lr_scheduler.LambdaLR, *args, **kwargs)

    @staticmethod
    def StepLR(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.StepLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.StepLR
        """
        return partial(tc.optim.lr_scheduler.StepLR, *args, **kwargs)

    @staticmethod
    def MultiStepLR(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.MultiStepLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.MultiStepLR
        """
        return partial(tc.optim.lr_scheduler.MultiStepLR, *args, **kwargs)

    @staticmethod
    def ExponentialLR(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.ExponentialLR`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.ExponentialLR
        """
        return partial(tc.optim.lr_scheduler.ExponentialLR, *args, **kwargs)

    @staticmethod
    def ReduceLROnPlateau(*args, **kwargs):
        """
        Wrapper class for :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
        """
        return partial(tc.optim.lr_scheduler.ReduceLROnPlateau, *args, **kwargs)
