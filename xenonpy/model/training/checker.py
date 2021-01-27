#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from collections import defaultdict
from pathlib import Path
from typing import Any, Union, Dict, Callable, Tuple

import joblib
import pandas as pd
import torch
from deprecated import deprecated
from torch.nn import Module

from xenonpy.model.training.base import BaseRunner

__all__ = ['Checker']


class Checker(object):
    """
    Check point.
    """

    class __SL:
        dump = torch.save
        load = torch.load

    def __init__(
            self,
            path: Union[Path, str] = None,
            *,
            increment: bool = False,
            device: Union[bool, str, torch.device] = 'cpu',
            default_handle: Tuple[Callable, str] = (joblib, '.pkl.z'),
    ):
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
        if path is None:
            path = Path().cwd().name
            self._path = Path.cwd() / path
        else:
            self._path = Path.cwd() / path
        if increment:
            i = 1
            while Path(f'{path}@{i}').exists():
                i += 1
            self._path = Path.cwd() / f'{path}@{i}'
        self._path.mkdir(parents=True, exist_ok=True)
        # self._path = self._path.resolve()
        self._device = BaseRunner.check_device(device)
        self._handle = default_handle

        self._files: Dict[str, str] = defaultdict(str)
        self._make_file_index()

    @classmethod
    @deprecated(
        'This method is rotten and will be removed in v1.0.0, use `Checker(<model path>)` instead')
    def load(cls, model_path):
        return cls(model_path)

    @property
    def path(self):
        return str(self._path)

    @property
    def files(self):
        return list(self._files.keys())

    @property
    def model_name(self):
        """

        Returns
        -------
        str
            Model name.
        """
        return self._path.name

    @property
    def model_structure(self):
        structure = self['model_structure']
        print(structure)
        return structure

    @property
    def training_info(self):
        return self['training_info']

    @property
    def describe(self):
        """
        Description for this model.

        Returns
        -------
        dict
            Description.
        """
        return self['describe']

    @property
    def model(self):
        """

        Returns
        -------
        model: :class:`torch.nn.Module`
            A pytorch model.
        """
        if (self._path / 'model.pth.m').exists():
            model = torch.load(str(self._path / 'model.pth.m'), map_location=self._device)
            state = self.final_state

            if state is not None:
                try:
                    model.load_state_dict(state)
                except torch.nn.modules.module.ModuleAttributeError:
                    # pytorch 1.6.0 compatability
                    for _, m in model.named_modules():
                        m._non_persistent_buffers_set = set()
                    model.load_state_dict(state)
                return model
            else:
                return model
        return None

    @model.setter
    def model(self, model: Module):
        """
        Set a model instance.

        Parameters
        ----------
        model: :class:`torch.nn.Module`
            Pytorch model instance.
        """
        if isinstance(model, Module):
            self(model=model)
            self.init_state = model.state_dict()
            self(model_structure=str(model))
        else:
            raise TypeError(f'except `torch.nn.Module` object but got {type(model)}')

    @property
    @deprecated('This property is rotten and will be removed in v1.0.0, use `checker.model` instead'
               )
    def trained_model(self):
        if (self._path / 'trained_model.@1.pkl.z').exists():
            return torch.load(str(self._path / 'trained_model.@1.pkl.z'), map_location=self._device)
        else:
            tmp = self.final_state
            if tmp is not None:
                model: torch.nn.Module = self.model
                model.load_state_dict(tmp)
                return model
        return None

    @property
    def model_class(self):
        if (self._path / 'model_class.pkl.z').exists():
            return self['model_class']
        return None

    @property
    def model_params(self):
        if (self._path / 'model_params.pkl.z').exists():
            return self['model_params']
        return None

    @property
    def init_state(self):
        if (self._path / 'init_state.pth.s').exists():
            return torch.load(str(self._path / 'init_state.pth.s'), map_location=self._device)
        return None

    @init_state.setter
    def init_state(self, state: OrderedDict):
        if not isinstance(state, OrderedDict) or not state:
            raise TypeError
        for v in state.values():
            if not isinstance(v, torch.Tensor):
                raise TypeError()
        self((Checker.__SL, '.pth.s'), init_state=state)

    @property
    def final_state(self):
        if (self._path / 'final_state.pth.s').exists():
            return torch.load(str(self._path / 'final_state.pth.s'), map_location=self._device)
        return None

    @final_state.setter
    def final_state(self, state: OrderedDict):
        if not isinstance(state, OrderedDict) or not state:
            raise TypeError
        for v in state.values():
            if not isinstance(v, torch.Tensor):
                raise TypeError()
        self((Checker.__SL, '.pth.s'), final_state=state)

    def _make_file_index(self):

        for f in [
                f for f in self._path.iterdir()
                if f.match('*.pkl.*') or f.match('*.pd.*') or f.match('*.pth.*')
        ]:
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
        if fp == '':
            return None
        fp_ = Path(fp)
        if not fp_.exists():
            return None
        patten = fp_.name.split('.')[-2]
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
        if name == 'checkpoints':
            sub_set = self.__class__(self._path / name, increment=False, device=self._device)
            setattr(self, f'{name}', sub_set)
            return sub_set

        raise AttributeError(f'no such attribute named {name}')

    def __getitem__(self, item):

        if isinstance(item, str):
            return self._load_data(item, self._handle[0])
        else:
            raise KeyError(f'{item}')

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
        self.checkpoints((Checker.__SL, '.pth.s'), **kwargs)

    def __repr__(self):
        cont_ls = ['<{}> includes:'.format(self.__class__.__name__)]

        for k, v in self._files.items():
            cont_ls.append('"{}": {}'.format(k, v))

        return '\n'.join(cont_ls)
