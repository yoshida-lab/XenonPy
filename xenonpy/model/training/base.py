#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import defaultdict
from inspect import signature
from typing import Iterable
from typing import Union, Tuple, DefaultDict, Dict

import torch
from sklearn.base import BaseEstimator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from xenonpy.model.training.extension.base import BaseExtension
from xenonpy.utils import TimedMetaClass, camel_to_snake

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

    def __init__(self, cuda: Union[bool, str, torch.device] = False):
        self._device = self.check_cuda(cuda)
        self._extensions: BaseRunner.T_Extension_Dict = defaultdict(
            lambda: (None, {}))

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

    def input_proc(self, x_in, y_in=None, train=True):
        for (ext, injects) in self._extensions.values():
            x_in, y_in = ext.input_proc(x_in=x_in, y_in=y_in, train=train,
                                        **{k: self._extensions[k][0] for k in injects['input_proc']})
        if y_in is None:
            return x_in
        return x_in, y_in

    def output_proc(self, y_pred, train=True):
        for (ext, injects) in self._extensions.values():
            y_pred = ext.output_proc(y_pred=y_pred, train=train,
                                     **{k: self._extensions[k][0] for k in injects['output_proc']})
        return y_pred

    def _before_proc(self, train=True):
        for (ext, injects) in self._extensions.values():
            ext.before_proc(train=train, **{k: self._extensions[k][0] for k in injects['before_proc']})

    def _step_forward(self, step_info):
        for (ext, injects) in self._extensions.values():
            ext.step_forward(step_info, **{k: self._extensions[k][0] for k in injects['step_forward']})

    def _after_proc(self, train=True):
        for (ext, injects) in self._extensions.values():
            ext.after_proc(train=train, **{k: self._extensions[k][0] for k in injects['after_proc']})

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
            return [p.name for p in sig.parameters.values() if p.kind == p.KEYWORD_ONLY and p.default is None]

        # merge exts to named_exts
        for ext in exts:
            name = ext.__class__.__name__
            named_exts[name] = ext

        for k, v in named_exts.items():
            k = camel_to_snake(k)
            v.runner = self

            methods = ['before_proc', 'input_proc', 'step_forward', 'output_proc', 'after_proc']
            dependencies = [_get_keyword_params(getattr(v, m)) for m in methods]
            dependency_inject = {k: v for k, v in zip(methods, dependencies)}

            self._extensions[k] = (v, dependency_inject)
