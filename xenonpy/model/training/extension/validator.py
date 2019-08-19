#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from collections import OrderedDict
from typing import Callable, Any, Dict

import numpy as np

from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension

__all__ = ['Validator']


class Validator(BaseExtension):

    def __init__(self, *,
                 metrics_func: Callable[[Any, Any], Dict],
                 early_stopping: int = None,
                 trace_order: int = 1,
                 **trace_metrics: Dict[str, float]
                 ):
        self.metrics_func = metrics_func
        self.patience = early_stopping + 1 if early_stopping is not None else None
        self._count = early_stopping
        self.order = trace_order

        self.trace = {}
        self.trace_order = trace_order
        self.trace_metrics = trace_metrics
        self._set_trace(trace_metrics, trace_order)

        self.from_dataset = False
        self.train_loss = np.inf

    def _set_trace(self, trace_metrics: dict, trace_order: int):
        for name, target in trace_metrics.items():
            self.trace[name] = (target, [np.inf] * trace_order)

    def on_reset(self) -> None:
        self._set_trace(self.trace_metrics, self.trace_order)

    def before_proc(self, trainer: Trainer) -> None:
        x_val, y_val = trainer.x_val, trainer.y_val
        val_dataset = trainer.validate_dataset

        if x_val is None and y_val is None and val_dataset is not None:
            self.from_dataset = True
        elif x_val is None or y_val is None:
            raise RuntimeError('no data for validation')

    def step_forward(self, trainer: Trainer, step_info: OrderedDict) -> None:
        if self.from_dataset:
            y_preds, y_trues = trainer.predict(dataset=trainer.validate_dataset)
        else:
            y_preds, y_trues = trainer.predict(trainer.x_val, trainer.y_val)

        train_loss = step_info[trainer.loss_type]
        if train_loss < self.train_loss:
            self.train_loss = train_loss
            self._count = self.patience

        metrics = self.metrics_func(y_trues, y_preds)
        for name, (target, current) in self.trace.items():
            if name in metrics:
                score = np.abs(metrics[name] - target)
                if score < current[-1]:
                    current.append(score)
                    current.sort()
                    current.pop()
                    self._count = self.patience
                    if self.order == 1:
                        trainer.set_checkpoint(name)
                    else:
                        index = current.index(score) + 1
                        for i in range(self.order, index, -1):
                            if f'{name}:{i - 1}' in trainer.checkpoints:
                                trainer.checkpoints[f'{name}:{i}'] = trainer.checkpoints[f'{name}:{i - 1}']
                        trainer.set_checkpoint(f'{name}:{index}')

        if self.patience is not None:
            self._count -= 1
            if self._count == 0:
                trainer.early_stop(
                    f'no improvement for {[k for k in self.trace]} since the last {self.patience} iterations, '
                    f'finish training at iteration {trainer.total_iterations}')

        step_info.update({f'val_{k}': v for k, v in metrics.items()})
