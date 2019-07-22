#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Union, Tuple, Callable, Any, Dict

from xenonpy.model.training.extension.base import BaseExtension

__all__ = ['Validator']


class Validator(BaseExtension):

    def __init__(self,
                 x_val: Union[Any, Tuple[Any]],
                 y_val: Any,
                 metrics_func: Callable[[Any, Any], Dict],
                 ):
        super().__init__()

        self.x_val = x_val
        self.y_val = y_val
        self.metrics_func = metrics_func

    def before_proc(self, **kwargs) -> None:
        self.x_val = self.runner.input_proc(x_in=self.x_val, train=False)

    def step_forward(self, step_info, **kwargs) -> None:
        y_pred = self.runner.predict(self.x_val)
        metrics = self.metrics_func(y_pred, self.y_val)
        step_info.update({f'val_{k}': v for k, v in metrics.items()})
