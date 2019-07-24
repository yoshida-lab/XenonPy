#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from functools import wraps
from inspect import signature
from typing import Iterable
from typing import Tuple, Any
from typing import Union, Dict

import torch
from sklearn.base import BaseEstimator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from xenonpy.utils import TimedMetaClass, camel_to_snake

__all__ = ['BaseRunner', 'BaseLRScheduler', 'BaseOptimizer', 'BaseExtension']


def _none_return_wrap(func):
    @wraps(func)
    def _func(self, *args_, **kwargs_):
        func(self, *args_, **kwargs_)
        return args_ if len(args_) > 1 else args_[0]

    return _func


class BaseExtension(object):
    def before_proc(self, **dependence) -> None:
        pass

    @_none_return_wrap
    def input_proc(self, x_in, y_in, **dependence) -> Tuple[Any]:
        pass

    def step_forward(self, step_info: OrderedDict, **dependence) -> None:
        pass

    @_none_return_wrap
    def output_proc(self, y_pred, **dependence) -> Any:
        pass

    def after_proc(self, **dependence) -> None:
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
    T_Extension_Dict = Dict[str, Tuple[BaseExtension, Dict[str, list]]]

    def __init__(self, cuda: Union[bool, str, torch.device] = False):
        self._device = self.check_cuda(cuda)
        self._extensions: BaseRunner.T_Extension_Dict = {}

    @staticmethod
    def check_cuda(cuda: Union[bool, str, torch.device]) -> torch.device:
        if isinstance(cuda, bool):
            if cuda:
                if torch.cuda.is_available():
                    return torch.device('cuda')
                else:
                    raise RuntimeError('could not use CUDA on this machine')
            else:
                return torch.device('cpu')

        if isinstance(cuda, str):
            if 'cuda' in cuda:
                if torch.cuda.is_available():
                    return torch.device(cuda)
                else:
                    raise RuntimeError('could not use CUDA on this machine')
            elif 'cpu' in cuda:
                return torch.device('cpu')
            else:
                raise RuntimeError('wrong device identifier'
                                   'see also: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device')

        if isinstance(cuda, torch.device):
            return cuda

    def input_proc(self, x_in, y_in=None, training=True):
        for (ext, injects) in self._extensions.values():
            kwargs = {k: self._extensions[k][0] for k in injects['input_proc'] if k in self._extensions}
            if 'trainer' in injects['input_proc']:
                kwargs.update(trainer=self)
            if 'training' in injects['input_proc']:
                kwargs.update(training=training)
            x_in, y_in = ext.input_proc(x_in, y_in, **kwargs)
        if y_in is None:
            return x_in
        return x_in, y_in

    def output_proc(self, y_pred, training=True):
        for (ext, injects) in self._extensions.values():
            kwargs = {k: self._extensions[k][0] for k in injects['output_proc'] if k in self._extensions}
            if 'trainer' in injects['output_proc']:
                kwargs.update(trainer=self)
            if 'training' in injects['output_proc']:
                kwargs.update(training=training)
            y_pred = ext.output_proc(y_pred, **kwargs)
        return y_pred

    def _before_proc(self, training=True):
        for (ext, injects) in self._extensions.values():
            kwargs = {k: self._extensions[k][0] for k in injects['before_proc'] if k in self._extensions}
            if 'trainer' in injects['before_proc']:
                kwargs.update(trainer=self)
            if 'training' in injects['before_proc']:
                kwargs.update(training=training)
            ext.before_proc(**kwargs)

    def _step_forward(self, step_info):
        for (ext, injects) in self._extensions.values():
            kwargs = {k: self._extensions[k][0] for k in injects['step_forward'] if k in self._extensions}
            if 'trainer' in injects['step_forward']:
                kwargs.update(trainer=self)
            ext.step_forward(step_info, **kwargs)

    def _after_proc(self, training=True):
        for (ext, injects) in self._extensions.values():
            kwargs = {k: self._extensions[k][0] for k in injects['after_proc'] if k in self._extensions}
            if 'trainer' in injects['after_proc']:
                kwargs.update(trainer=self)
            if 'training' in injects['after_proc']:
                kwargs.update(training=training)
            ext.after_proc(**kwargs)

    def extend(self, *extension: BaseExtension):
        """
        Add training extensions to trainer.

        Parameters
        ----------
        extension: BaseExtension
            Extension.

        """

        def _get_keyword_params(func) -> list:
            sig = signature(func)
            return [p.name for p in sig.parameters.values() if
                    p.kind == p.KEYWORD_ONLY and p.default is p.empty]

        # merge exts to named_exts
        for ext in extension:
            name = camel_to_snake(ext.__class__.__name__)
            methods = ['before_proc', 'input_proc', 'step_forward', 'output_proc', 'after_proc']
            dependencies = [_get_keyword_params(getattr(ext, m)) for m in methods]
            dependency_inject = {k: v for k, v in zip(methods, dependencies)}

            self._extensions[name] = (ext, dependency_inject)
