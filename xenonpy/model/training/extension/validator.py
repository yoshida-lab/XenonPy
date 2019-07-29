#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Callable, Any, Dict

import numpy as np

from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension

__all__ = ['Validator']


class Validator(BaseExtension):

    def __init__(self, *,
                 metrics_func: Callable[[Any, Any], Dict],
                 early_stopping: float = None,
                 **trace_metrics: Dict[str, float]
                 ):
        self.metrics_func = metrics_func
        self.patience = early_stopping
        self._count = early_stopping

        self.trace = {}
        for name, target in trace_metrics.items():
            self.trace[name] = (target, np.inf)

        self.from_dataset = False

    def before_proc(self, *, trainer: Trainer) -> None:
        x_val, y_val = trainer.x_val, trainer.y_val
        val_dataset = trainer.validate_dataset

        if x_val is None and y_val is None and val_dataset is not None:
            self.from_dataset = True
        elif x_val is None or y_val is None:
            raise RuntimeError('no data for validation')

    def step_forward(self, step_info, *, trainer: Trainer) -> None:
        if self.from_dataset:
            y_preds, y_trues = trainer.predict(dataset=trainer.validate_dataset)
        else:
            y_preds, y_trues = trainer.predict(trainer.x_val, trainer.y_val)

        metrics = self.metrics_func(y_preds, y_trues)
        for name, (target, current) in self.trace.items():
            if name in metrics:
                score = np.abs(metrics[name] - target)
                if score < current:
                    self.trace[name] = (target, score)
                    self._count = self.patience
                    trainer.snapshot(name, target=target)
                else:
                    if self.patience is not None:
                        self._count -= 1
                        if self._count == 0:
                            trainer.early_stopping = True

        step_info.update({f'val_{k}': v for k, v in metrics.items()})
