#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from copy import deepcopy
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
                 model_class: Callable = None,
                 reconstruct_params: Union[list, dict] = None,
                 increment=False,
                 sync_training_step=False,
                 save_optimizer_state=False,
                 **describe: Any):
        self._model_class: Callable = model_class
        self._reconstruct_params: Union[list, dict] = reconstruct_params
        self.save_optimizer_state = save_optimizer_state
        self.sync_training_step = sync_training_step
        self.path = path
        self._increment = increment
        self._describe = describe
        self._describe_ = None
        self._checker: Union[Checker, None] = None
        self._tmp_args: list = []
        self._tmp_kwargs: dict = {}

    @property
    def describe(self):
        return self._checker.describe

    @property
    def path(self):
        return str(self._path)

    @path.setter
    def path(self, path: Union[Path, str]):
        if path == '.':
            path = Path(path).resolve()
            self._path = path / path.name
        else:
            self._path = Path(path).resolve()

    @property
    def model_structure(self):
        return self._checker.model_structure

    def get_checkpoint(self, id_: str = None):
        if id_ is not None:
            return self._checker.checkpoints[id_]
        return self._checker.checkpoints.files

    def __call__(self, handle: Any = None, **kwargs: Any):
        if self._checker is None:
            raise RuntimeError('calling of this method only after the model training')
        self._checker(handle=handle, **kwargs)

    def __getitem__(self, item):
        return self._checker[item]

    def on_checkpoint(self, checkpoint: Trainer.checkpoint_tuple, trainer: Trainer) -> None:
        key = checkpoint.id.replace(':', '_')
        value = deepcopy(checkpoint._asdict())
        if not self.save_optimizer_state:
            del value['optimizer_state']
        # print(self._checker.model_name, key)
        self._checker.set_checkpoint(**{key: value})
        if self.sync_training_step and trainer.training_info is not None:
            self._checker(
                training_info=trainer.training_info,
            )

    def before_proc(self, trainer: Trainer) -> None:
        self._checker = Checker(self._path, increment=self._increment)
        if self._model_class is not None:
            self._checker(model_class=self._model_class)
        if self._reconstruct_params is not None:
            self._checker(init_params=self._reconstruct_params)
        self._checker.model = trainer.model
        self._describe_ = dict(
            python=py_ver,
            system=sys_ver(),
            numpy=np.__version__,
            torch=torch.__version__,
            xenonpy=__version__,
            device=str(trainer.device),
            start=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
            finish='N/A',
            time_elapsed='N/A',
            **self._describe,
        )
        self._checker(describe=self._describe_)

    def after_proc(self, trainer: Trainer) -> None:
        self._describe_.update(
            finish=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
            time_elapsed=str(timedelta(seconds=trainer.timer.elapsed)))
        self._checker.final_state = trainer.model.state_dict()
        self._checker(
            training_info=trainer.training_info,
            describe=self._describe_,
        )
