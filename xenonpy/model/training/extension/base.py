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
        ret = func(self, *args_, **kwargs_)
        if ret is None:
            return args_
        return ret

    return _func


class BaseExtension(object):
    def __init__(self):
        self.trainer = None

    def before_train(self, **kwargs) -> None:
        pass

    @_none_return_wrap
    def input_proc(self, x, y=None, **kwargs) -> Tuple[Any, Any]:
        pass

    def step_forward(self, step_info: OrderedDict, **kwargs) -> None:
        pass

    @_none_return_wrap
    def output_proc(self, y_pred, **kwargs) -> Any:
        pass

    def after_train(self, **kwargs) -> None:
        pass
