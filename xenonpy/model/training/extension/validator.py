#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
import numpy as np

from collections import OrderedDict
from typing import Callable, Any, Dict, Union

from xenonpy.model.utils import regression_metrics, classification_metrics
from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension

__all__ = ['Validator']


class Validator(BaseExtension):
    """
    Validator extension
    """

    regress = 'regress'
    classify = 'classify'

    def __init__(self,
                 metrics_func: Union[str, Callable[[Any, Any], Dict]],
                 *,
                 each_iteration: bool = True,
                 early_stopping: int = None,
                 trace_order: int = 1,
                 **trace_criteria: Dict[str, float]):
        """

        Parameters
        ----------
        metrics_func
            Function for metrics calculation.
            If ``str``, should be ``regress`` or ``classify`` which to specify the training types.
            See :py:func:`xenonpy.model.utils.classification_metrics` and
            :py:func:`xenonpy.model.utils.regression_metrics` to know the details.
            you can also give the calculation function yourself.
        each_iteration
            If ``True``, validation will be executed every iteration.
            Otherwise, only validate at each epoch done.
            Default ``True``.
        early_stopping
            Set patience condition of early stopping condition.
            This condition trace criteria setting by ``trace_criteria`` parameter.
        trace_order
            How many ranks of ``trace_criteria`` will be saved as checkpoint.
            Checkpoint name follow the format ``criterion_rank``, e.g. ``mae_1``
        trace_criteria
            Validation criteria.
            Should follow this formation: ``criterion=target``, e.g ``mae=0, corr=1``.
            The names of criteria must be consistent with the output of``metrics_func``.
        """
        if metrics_func == 'regress':
            self.metrics_func = regression_metrics
        elif metrics_func == 'classify':
            self.metrics_func = classification_metrics
        else:
            self.metrics_func = metrics_func

        self.each_iteration = each_iteration
        self.patience = early_stopping + 1 if early_stopping is not None else None
        self._count = early_stopping
        self.order = trace_order

        self._epoch_count = 0
        self.trace = {}
        self.trace_order = trace_order
        self.trace_criteria = trace_criteria
        self._set_trace(trace_criteria, trace_order)

        self.from_dataset = False
        self.train_loss = np.inf

    def _set_trace(self, trace_metrics: dict, trace_order: int):
        for name, target in trace_metrics.items():
            self.trace[name] = (target, [np.inf] * trace_order)

    def on_reset(self) -> None:
        self._set_trace(self.trace_criteria, self.trace_order)

    def before_proc(self, trainer: Trainer) -> None:
        x_val, y_val = trainer.x_val, trainer.y_val
        val_dataset = trainer.validate_dataset

        if x_val is None and y_val is None and val_dataset is not None:
            self.from_dataset = True
        elif x_val is None or y_val is None:
            raise RuntimeError('no data for validation')

    def step_forward(self, trainer: Trainer, step_info: OrderedDict) -> None:
        def _validate():
            if self.from_dataset:
                y_preds, y_trues = trainer.predict(
                    dataset=trainer.validate_dataset)
            else:
                y_preds, y_trues = trainer.predict(trainer.x_val,
                                                   trainer.y_val)

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
                                if f'{name}_{i - 1}' in trainer.checkpoints:
                                    trainer.checkpoints[
                                        f'{name}_{i}'] = trainer.checkpoints[
                                            f'{name}_{i - 1}']
                            trainer.set_checkpoint(f'{name}_{index}')

            if self.patience is not None:
                self._count -= 1
                if self._count == 0:
                    trainer.early_stop(
                        f'no improvement for {[k for k in self.trace]} since the last {self.patience} iterations, '
                        f'finish training at iteration {trainer.total_iterations}'
                    )

            step_info.update({f'val_{k}': v for k, v in metrics.items()})

        if not self.each_iteration:
            epoch = step_info['i_epoch']
            if epoch > self._epoch_count:
                self._epoch_count = epoch
                _validate()
        else:
            _validate()
