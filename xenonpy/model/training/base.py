#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import defaultdict
from inspect import signature
from typing import Iterable
from typing import Union, Tuple, DefaultDict, Dict

from sklearn.base import BaseEstimator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from xenonpy.model.training.extension.base import BaseExtension
from xenonpy.utils import TimedMetaClass

__all__ = ['BaseRunner', 'BaseLRScheduler', 'BaseOptimizer']


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
    T_Extension_Dict = DefaultDict[str, Tuple[Union[BaseExtension, None], Dict[str, list]]]

    def __init__(self):
        # init container
        self._extensions: BaseRunner.T_Extension_Dict = defaultdict(
            lambda: (None, {}))

    def input_proc(self, x_in, y_in=None):
        for ext, injects in self._extensions:
            x_in, y_in = ext.before_train(x_in=x_in, y_in=y_in,
                                          **{k: self._extensions[k][0] for k in injects['input_proc']})
        return x_in, y_in

    def output_proc(self, y_pred):
        for ext, injects in self._extensions:
            y_pred = ext.output_proc(y_pred=y_pred, **{k: self._extensions[k][0] for k in injects['output_proc']})
        return y_pred

    def _before_train(self):
        for ext, injects in self._extensions:
            ext.before_train(**{k: self._extensions[k][0] for k in injects['before_train']})

    def _step_forward(self, step_info):
        for ext, injects in self._extensions:
            ext.step_forward(step_info, **{k: self._extensions[k][0] for k in injects['step_forward']})

    def _after_train(self):
        for ext, injects in self._extensions:
            ext.after_train(**{k: self._extensions[k][0] for k in injects['after_train']})

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

            methods = ['before_train', 'input_proc', 'step_forward', 'output_proc', 'after_train']
            dependencies = [_get_keyword_params(getattr(v, m)) for m in methods]
            dependency_inject = {k: v for k, v in zip(methods, dependencies)}

            self._extensions[k] = (v, dependency_inject)
