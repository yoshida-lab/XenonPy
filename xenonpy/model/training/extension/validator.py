#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Union, Tuple, Callable, Any, Dict

import numpy as np
import torch

from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension

__all__ = ['Validator']


class Validator(BaseExtension):

    def __init__(self,
                 x_val: Union[Any, Tuple[Any]],
                 y_val: Any,
                 metrics_func: Callable[[Any, Any], Dict],
                 *trace_metrics: Tuple[str, float]
                 ):
        self.x_val = x_val
        self.y_val = y_val
        self.metrics_func = metrics_func

        self.trace = {}
        for (name, target) in trace_metrics:
            self.trace[name] = (target, np.inf)

    def before_proc(self, *, trainer: Trainer) -> None:
        self.x_val = trainer.input_proc(x_in=self.x_val, training=False)

    def step_forward(self, step_info, *, trainer: Trainer) -> None:
        y_pred = trainer.predict(self.x_val)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().numpy()
        metrics = self.metrics_func(y_pred, self.y_val)
        for name, (target, current) in self.trace.items():
            if name in metrics:
                score = np.abs(metrics[name] - target)
                if score < current:
                    self.trace[name] = (target, score)
                    trainer.snapshot(name, target=target)

        step_info.update({f'val_{k}': v for k, v in metrics.items()})
