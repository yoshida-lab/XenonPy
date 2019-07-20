#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from typing import Iterable

from torch.optim import Optimizer


class BaseExtension(object):
    def run(self, step_info: OrderedDict, trainer):
        raise NotImplementedError()


class BaseOptimizer(object):
    def __init__(self, optimizer, **kwargs):
        self._kwargs = kwargs
        self._optimizer = optimizer

    def __call__(self, params: Iterable):
        """

        Parameters
        ----------
        params: Iterable
            iterable of parameters to optimize or dicts defining
            parameter groups

        Returns
        -------
        optimizeer: Optimizer
        """
        return self._optimizer(params, **self._kwargs)


class BaseLRScheduler(object):
    def __init__(self, lr_scheduler, **kwargs):
        self._kwargs = kwargs
        self._lr_scheduler = lr_scheduler

    def __call__(self, optimizer: Optimizer):
        """

        Parameters
        ----------
        optimizer: Optimizer
            Wrapped optimizer.

        Returns
        -------

        """
        return self._lr_scheduler(optimizer, **self._kwargs)
