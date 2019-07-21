#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict, defaultdict
from inspect import signature
from typing import Iterable
from typing import Union, Tuple, DefaultDict, List

from sklearn.base import BaseEstimator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from xenonpy.utils import TimedMetaClass


class BaseExtension(object):
    def __init__(self):
        self.trainer = None

    def init(self):
        pass

    def pre_process(self, x, y=None):
        return x, y

    def step_forward(self, step_info: OrderedDict, **kwargs):
        raise NotImplementedError()

    def post_process(self, y_pred):
        return y_pred

    def final(self):
        pass


class BaseOptimizer(object):
    def __init__(self, optimizer, **kwargs):
        self._kwargs = kwargs
        self._optimizer = optimizer

    def __call__(self, params: Iterable) -> Optimizer:
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

    def __call__(self, optimizer: Optimizer) -> _LRScheduler:
        """

        Parameters
        ----------
        optimizer: Optimizer
            Wrapped optimizer.

        Returns
        -------

        """
        return self._lr_scheduler(optimizer, **self._kwargs)


class BaseRunner(BaseEstimator, metaclass=TimedMetaClass):
    def __init__(self):
        # init container
        self._extensions: DefaultDict[str, Tuple[Union[BaseExtension, None], List]] = defaultdict(lambda: (None, []))

    def extend(self, *exts: BaseExtension, **named_exts: BaseExtension):
        """
        Add training extensions to trainer.

        Parameters
        ----------
        exts: BaseExtension
            Extension.

        """

        def _get_keyword_params(func) -> list:
            sig = signature(func)
            return [p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY]

        # merge exts to named_exts
        for ext in exts:
            name = ext.__class__.__name__
            named_exts[name] = ext

        for k, v in named_exts.items():
            v.trainer = self
            v.init()
            self._extensions[k] = (v, _get_keyword_params(v.step_forward))
