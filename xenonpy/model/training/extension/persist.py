#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from datetime import datetime, timedelta
from pathlib import Path
from platform import version as sys_ver
from sys import version as py_ver
from typing import Union, Callable, Any

import numpy as np
import torch

from xenonpy import __version__
from xenonpy.model.training import Trainer
from xenonpy.model.training.base import BaseExtension
from xenonpy.model.utils import Checker

__all__ = ['Persist']


class Persist(BaseExtension):

    def __init__(self,
                 path: Union[Path, str] = '.',
                 *,
                 init_params: Union[list, dict] = None,
                 model_class: Callable = None,
                 increment=False,
                 save_optimizer_state=False,
                 sync_training_step=False,
                 **describe: Any):
        self.save_optimizer_state = save_optimizer_state
        self.sync_training_step = sync_training_step
        self.checker = Checker(path, increment=increment)
        self.describe = dict(
            python=py_ver,
            system=sys_ver(),
            numpy=np.__version__,
            torch=torch.__version__,
            xenonpy=__version__,
            device='N/A',
            start='N/A',
            finish='N/A',
            time_elapsed='N/A',
            **describe,
        )
        self.checker(describe=self.describe)
        if model_class is not None:
            self.checker(model_class=model_class)
        if init_params is not None:
            self.checker(init_params=init_params)

    def __call__(self, *args: Any, **kwargs: Any):
        self.checker(*args, **kwargs)

    def on_checkpoint(self, checkpoint: Trainer.checkpoint_tuple, trainer: Trainer) -> None:
        key = checkpoint.id.replace(':', '_')
        value = checkpoint._asdict()
        if not self.save_optimizer_state:
            del value['optimizer_state']
        self.checker.set_checkpoint(**{key: value})
        if self.sync_training_step:
            self.checker(
                training_info=trainer.training_info,
            )

    def before_proc(self, trainer: Trainer) -> None:
        self.describe.update(
            device=str(trainer.device),
            start=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
        )
        self.checker(describe=self.describe)
        self.checker.model = trainer.model

    def after_proc(self, trainer: Trainer) -> None:
        self.describe.update(
            finish=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
            time_elapsed=str(timedelta(seconds=trainer.timer.elapsed)))
        self.checker.final_state = trainer.model.state_dict()
        self.checker(
            training_info=trainer.training_info,
            describe=self.describe,
        )
