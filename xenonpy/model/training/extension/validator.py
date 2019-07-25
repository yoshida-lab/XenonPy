#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Union, Tuple, Callable, Any, Dict

from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension

__all__ = ['Validator']


class Validator(BaseExtension):

    def __init__(self,
                 x_val: Union[Any, Tuple[Any]],
                 y_val: Any,
                 metrics_func: Callable[[Any, Any], Dict],
                 trace_metrics: Union[list, str] = None
                 ):
        self.x_val = x_val
        self.y_val = y_val
        self.metrics_func = metrics_func

        self.trace = {}
        if trace_metrics is not None:
            if isinstance(trace_metrics, str):
                self.trace[trace_metrics] = None
            elif isinstance(trace_metrics, list):
                for t in trace_metrics:
                    self.trace[t] = None
            else:
                raise RuntimeError()

    def before_proc(self, *, trainer: Trainer) -> None:
        self.x_val = trainer.input_proc(x_in=self.x_val, training=False)

    def step_forward(self, step_info, *, trainer: Trainer) -> None:
        y_pred = trainer.predict(self.x_val)
        metrics = self.metrics_func(y_pred, self.y_val)
        for t in self.trace.keys():
            k = f'val_{t}'
            if k in metrics:
                v = metrics[k]
                if self.trace[t] is None:
                    self.trace[t] = v
                    trainer.check_point()
                else:
                    pass
                # fixme: need implementation

        step_info.update({f'val_{k}': v for k, v in metrics.items()})
