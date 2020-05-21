#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from platform import version as sys_ver
from sys import version as py_ver
from typing import Union, Callable, Any

import numpy as np
import torch

from xenonpy import __version__
from xenonpy.model.training import Trainer, Checker
from xenonpy.model.training.base import BaseExtension

__all__ = ['Persist']


class Persist(BaseExtension):
    """
    Trainer extension for data persistence
    """

    def __init__(self,
                 path: Union[Path, str] = None,
                 *,
                 model_class: Callable = None,
                 model_params: Union[tuple, dict, any] = None,
                 increment=False,
                 sync_training_step=False,
                 **describe: Any):
        """

        Parameters
        ----------
        path
            Path for model saving.
        model_class
            A factory function for model reconstructing.
            In most case this is the model class inherits from :class:`torch.nn.Module`
        model_params
            The parameters for model reconstructing.
            This can be anything but in general this is a dict which can be used as kwargs parameters.
        increment
            If ``True``, dir name of path will be decorated with a auto increment number,
            e.g. use ``model_dir@1`` for ``model_dir``.
        sync_training_step
            If ``True``, will save ``trainer.training_info`` at each iteration.
            Default is ``False``, only save ``trainer.training_info`` at each epoch.
        describe:
            Any other information to describe this model.
            These information will be saved under model dir by name ``describe.pkl.z``.
        """
        self._model_class: Callable = model_class
        self._model_params: Union[list, dict] = model_params
        self.sync_training_step = sync_training_step
        self._increment = increment
        self._describe = describe
        self._describe_ = None
        self._checker: Union[Checker, None] = None
        self._tmp_args: list = []
        self._tmp_kwargs: dict = {}
        self._epoch_count = 0
        self.path = path

    @property
    def describe(self):
        if self._checker is None:
            raise ValueError('can not access property `describe` before training')
        return self._checker.describe

    @property
    def path(self):
        if self._checker is None:
            raise ValueError('can not access property `path` before training')
        return str(self._checker.path)

    @path.setter
    def path(self, path: Union[Path, str]):
        if self._checker is not None:
            raise ValueError('can not reset property `path` after training')
        self._path = path

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
        key = checkpoint.id
        value = deepcopy(checkpoint._asdict())
        self._checker.set_checkpoint(**{key: value})

    def step_forward(self, step_info: OrderedDict, trainer: Trainer) -> None:
        if self.sync_training_step:
            training_info = trainer.training_info
            if training_info is not None:
                self._checker(training_info=training_info)
        else:
            epoch = step_info['i_epoch']
            if epoch > self._epoch_count:
                training_info = trainer.training_info
                if training_info is not None:
                    self._epoch_count = epoch
                    self._checker(training_info=training_info)

    def before_proc(self, trainer: Trainer) -> None:
        self._checker = Checker(self._path, increment=self._increment)
        if self._model_class is not None:
            self._checker(model_class=self._model_class)
        if self._model_params is not None:
            self._checker(model_params=self._model_params)
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
        self._describe_.update(finish=datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                               time_elapsed=str(timedelta(seconds=trainer.timer.elapsed)))
        self._checker.final_state = trainer.model.state_dict()
        self._checker(
            training_info=trainer.training_info,
            describe=self._describe_,
        )
