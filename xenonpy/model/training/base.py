#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from inspect import signature
from typing import Iterable
from typing import Tuple, Any
from typing import Union, Dict

import torch
from sklearn.base import BaseEstimator
from torch.optim import Optimizer  # noqa
from torch.optim.lr_scheduler import _LRScheduler  # noqa

from xenonpy.utils import TimedMetaClass, camel_to_snake

__all__ = ['BaseRunner', 'BaseLRScheduler', 'BaseOptimizer', 'BaseExtension']


class BaseExtension(object):
    def before_proc(self, *dependence) -> None:
        pass

    def input_proc(self, x_in, y_in, *dependence) -> Tuple[Any, Any]:
        return x_in, y_in

    def step_forward(self, *dependence) -> None:
        pass

    def output_proc(self, y_pred, y_true, *dependence) -> Tuple[Any, Any]:
        return y_pred, y_true

    def after_proc(self, *dependence) -> None:
        pass

    def on_reset(self, *dependence) -> None:
        pass

    def on_checkpoint(self, *dependence) -> None:
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
        optimizer: Optimizer
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
        self._device = self.check_device(cuda)
        self._extensions: BaseRunner.T_Extension_Dict = {}

    @property
    def cuda(self):
        return self._device

    @staticmethod
    def check_device(cuda: Union[bool, str, torch.device]) -> torch.device:
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
                raise RuntimeError(
                    'wrong device identifier'
                    'see also: https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device'
                )

        if isinstance(cuda, torch.device):
            return cuda

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, v):
        self._device = self.check_device(v)

    def _make_inject(self, injects, kwargs):
        _kwargs = {
            k: self._extensions[k][0]
            for k in injects if k in self._extensions
        }
        _kwargs.update({k: kwargs[k] for k in injects if k in kwargs})
        return _kwargs

    def input_proc(self, x_in, y_in=None, **kwargs):
        for (ext, injects) in self._extensions.values():
            _kwargs = self._make_inject(injects['input_proc'], kwargs)
            x_in, y_in = ext.input_proc(x_in, y_in, **_kwargs)
        return x_in, y_in

    def output_proc(self, y_pred, y_true=None, **kwargs):
        for (ext, injects) in self._extensions.values():
            _kwargs = self._make_inject(injects['output_proc'], kwargs)
            y_pred, y_true = ext.output_proc(y_pred, y_true, **_kwargs)
        return y_pred, y_true

    def _before_proc(self, **kwargs):
        for (ext, injects) in self._extensions.values():
            _kwargs = self._make_inject(injects['before_proc'], kwargs)
            ext.before_proc(**_kwargs)

    def _step_forward(self, **kwargs):
        for (ext, injects) in self._extensions.values():
            _kwargs = self._make_inject(injects['step_forward'], kwargs)
            ext.step_forward(**_kwargs)

    def _after_proc(self, **kwargs):
        for (ext, injects) in self._extensions.values():
            _kwargs = self._make_inject(injects['after_proc'], kwargs)
            ext.after_proc(**_kwargs)

    def _on_reset(self, **kwargs):
        for (ext, injects) in self._extensions.values():
            _kwargs = self._make_inject(injects['on_reset'], kwargs)
            ext.on_reset(**_kwargs)

    def _on_checkpoint(self, **kwargs):
        for (ext, injects) in self._extensions.values():
            _kwargs = self._make_inject(injects['on_checkpoint'], kwargs)
            ext.on_checkpoint(**_kwargs)

    def extend(self, *extension: BaseExtension) -> 'BaseRunner':
        """
        Add training extensions to trainer.

        Parameters
        ----------
        extension: BaseExtension
            Extension.

        """
        def _get_keyword_params(func) -> list:
            sig = signature(func)
            return [
                p.name for p in sig.parameters.values()
                if p.kind == p.POSITIONAL_OR_KEYWORD
            ]

        # merge exts to named_exts
        for ext in extension:
            name = camel_to_snake(ext.__class__.__name__)
            methods = [
                'before_proc', 'input_proc', 'step_forward', 'output_proc',
                'after_proc', 'on_reset', 'on_checkpoint'
            ]
            dependencies = [
                _get_keyword_params(getattr(ext, m)) for m in methods
            ]
            dependency_inject = {k: v for k, v in zip(methods, dependencies)}

            self._extensions[name] = (ext, dependency_inject)

        return self

    def remove_extension(self, *extension: str):
        for name in extension:
            if name in self._extensions:
                del self._extensions[name]

    def __getitem__(self, item):
        return self._extensions[item][0]
