#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from pathlib import Path
from typing import Any, Union, Dict, Callable, Tuple

import joblib
import pandas as pd
import torch

from xenonpy.model.training.base import BaseRunner

__all__ = ['Checker']


class Checker(object):
    """
    Check point.
    """

    class __SL:
        dump = torch.save
        load = torch.load

    def __init__(self,
                 path: Union[Path, str] = '.',
                 *,
                 increment: bool = False,
                 device: Union[bool, str, torch.device] = 'cpu',
                 default_handle: Tuple[Callable, str] = (joblib, '.pkl.z')):
        """
        Parameters
        ----------
        path: Union[Path, str]
            Dir path for data access. Can be ``Path`` or ``str``.
            Given a relative path will be resolved to abstract path automatically.
        increment : bool
            Set to ``True`` to prevent the potential risk of overwriting.
            Default ``False``.
        """
        path = Path(path).resolve()
        if increment:
            i = 1
            while Path(f'{path}@{i}').exists():
                i += 1
            path = f'{path}@{i}'
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._device = BaseRunner.check_device(device)
        self._handle = default_handle

        self._files: Dict[str, str] = {}
        self._make_file_index()

    @classmethod
    def load(cls, model_path):
        return cls(model_path)

    @property
    def path(self):
        return str(self._path)

    @property
    def model_name(self):
        return self._path.name

    @property
    def describe(self):
        """
        Model's description.
        This is a property with getter/setter.
        This action don't overwrite anything but add a new object.

        Returns
        -------
        dict
            Description.
        """
        return self.last('describe')

    @property
    def init_model(self):
        """
        Last appended initial model.
        This is a property with getter/setter.
        This action don't overwrite anything but add a new object.

        Returns
        -------
        model:
            The last appended model.
        """
        return torch.load(str(self._path / 'init_model.pth.m'), map_location=self._device)

    @init_model.setter
    def init_model(self, model):
        """
        Set initial model.
        This action don't overwrite but add a new object.

        Parameters
        ----------
        model: torch.nn.Module
        """
        if isinstance(model, torch.nn.Module):
            self(init_model=model)
        else:
            raise TypeError(f'except `torch.nn.Module` object but got {type(model)}')

    @property
    def trained_model(self):
        """
        Last appended pre-trained model.
        This is a property with getter/setter.
        This action don't overwrite anything but add a new object.

        Returns
        -------
        model:
            The last appended model.
        """
        return torch.load(str(self._path / 'trained_model.pth.m'), map_location=self._device)

    @trained_model.setter
    def trained_model(self, model):
        """
        Set pre-trained model

        Parameters
        ----------
        model: torch.nn.Module
        """
        if isinstance(model, torch.nn.Module):
            self(trained_model=model)
        else:
            raise TypeError(f'except `torch.nn.Module` object but got {type(model)}')

    def _make_file_index(self):

        for f in [f for f in self._path.iterdir() if f.match('*.pkl.*') or f.match('*.pd.*') or f.match('*.pth.*')]:
            # select data
            fn = '.'.join(f.name.split('.')[:-2])
            self._files[fn] = str(f)

    def _save_data(self, data: Any, filename: str, handle) -> str:
        if isinstance(data, pd.DataFrame):
            file = str(self._path / (filename + '.pd.xz'))
            self._files[filename] = file
            pd.to_pickle(data, file)
        elif isinstance(data, (torch.Tensor, torch.nn.Module)):
            file = str(self._path / (filename + '.pth.m'))
            self._files[filename] = file
            torch.save(data, file)
        else:
            file = str(self._path / (filename + handle[1]))
            self._files[filename] = file
            handle[0].dump(data, file)

        return file

    def _load_data(self, file: str, handle):
        fp = self._files[file]
        patten = Path(fp).name.split('.')[-2]
        if patten == 'pd':
            return pd.read_pickle(fp)
        if patten == 'pth':
            return torch.load(fp, map_location=self._device)
        if patten == 'pkl':
            return joblib.load(fp)
        else:
            return handle.load(fp)

    def __getattr__(self, name: str):
        """
        Return sub-dataset.

        Parameters
        ----------
        name: str
            Dataset name.

        Returns
        -------
        self
        """
        sub_set = self.__class__(self._path / name, increment=False, device=self._device)
        setattr(self, f'{name}', sub_set)
        return sub_set

    def __getitem__(self, item):

        if isinstance(item, str):
            return self._load_data(item, self._handle[0])
        else:
            KeyError()

    def __call__(self, handle=None, **named_data: Any):
        """
        Save data with or without name.
        Data with same name will not be overwritten.

        Parameters
        ----------
        handle: Tuple[Callable, str]
        named_data: dict
            Named data as k,v pair.

        """
        if handle is None:
            handle = self._handle

        for k, v in named_data.items():
            self._save_data(v, k, handle)

    def set_checkpoint(self, **kwargs):
        self.checkpoints((Checker.__SL, '.pth.m'), **kwargs)
