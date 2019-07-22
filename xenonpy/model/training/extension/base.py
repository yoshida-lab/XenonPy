#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from collections import OrderedDict
from functools import wraps
from typing import Tuple, Any

__all__ = ['BaseExtension']


def _none_return_wrap(func):
    @wraps(func)
    def _func(self, *args_, **kwargs_):
        func(self, *args_, **kwargs_)
        return args_ if len(args_) > 1 else args_[0]

    return _func


class BaseExtension(object):
    def __init__(self):
        self.runner = None

    def before_proc(self, train: bool = True) -> None:
        pass

    @_none_return_wrap
    def input_proc(self, x_in, y_in, train: bool = True) -> Tuple[Any]:
        pass

    def step_forward(self, step_info: OrderedDict) -> None:
        pass

    @_none_return_wrap
    def output_proc(self, y_pred, train: bool = True) -> Any:
        pass

    def after_proc(self, train: bool = True) -> None:
        pass
