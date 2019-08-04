#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Tuple, Callable, Any, Dict

import numpy as np
import torch

from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension

__all__ = ['Validator']


class Validator(BaseExtension):

    def __init__(self,
                 metrics_func: Callable[[Any, Any], Dict],
                 **trace_metrics: Tuple[str, float]
                 ):
        self.metrics_func = metrics_func

        self.trace = {}
        for name, target in trace_metrics.items():
            self.trace[name] = (target, np.inf)

    def step_forward(self, step_info, *, trainer: Trainer) -> None:
        x_val, y_val = trainer.x_val, trainer.y_val
        y_pred = trainer.predict(x_val)
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().numpy()
        metrics = self.metrics_func(y_pred, y_val)
        for name, (target, current) in self.trace.items():
            if name in metrics:
                score = np.abs(metrics[name] - target)
                if score < current:
                    self.trace[name] = (target, score)
                    trainer.snapshot(name, target=target)

        step_info.update({f'val_{k}': v for k, v in metrics.items()})
