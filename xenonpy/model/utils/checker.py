#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from functools import partial
from pathlib import Path
from typing import Union

import joblib
import torch

from xenonpy.datatools.storage import Storage

__all__ = ['Checker']


class Checker(Storage):
    """
    Check point.
    """

    class __SL(object):
        load = partial(torch.load, map_location=torch.device('cpu'))
        dump = torch.save

    def __init__(self, path: Union[Path, str], *, increment: bool = False):
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
        if increment:
            i = 1
            while Path(f'{path}@{i}').exists():
                i += 1
            path = f'{path}@{i}'
        super().__init__(path)

    @classmethod
    def load(cls, model_path):
        return cls(model_path)

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
        self._backend = self.__SL
        try:
            return self.last('init_model')
        except FileNotFoundError:
            return None
        finally:
            self._backend = joblib

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
            self._backend = self.__SL
            super().__call__(init_model=model)
            self._backend = joblib
        else:
            raise TypeError(
                'except `torch.nn.Module` object but got {}'.format(
                    type(model)))

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
        self._backend = self.__SL
        try:
            return self.last('trained_model')
        except FileNotFoundError:
            return None
        finally:
            self._backend = joblib

    @trained_model.setter
    def trained_model(self, model):
        """
        Set pre-trained model

        Parameters
        ----------
        model: torch.nn.Module
        """
        if isinstance(model, torch.nn.Module):
            self._backend = self.__SL
            super().__call__(trained_model=model)
            self._backend = joblib
        else:
            raise TypeError(
                'except `torch.nn.Module` object but got {}'.format(
                    type(model)))

    def __getitem__(self, item):
        if isinstance(item, int):
            self._backend = self.__SL
            try:
                cp = self.checkpoints_[item]
                model_state = cp['model_state']
                del cp['model_state']
                return model_state, cp
            except IndexError:
                return None
            finally:
                self._backend = joblib
        raise TypeError('except int as checkpoint index but got {}'.format(
            type(item)))

    def __call__(self, **kwargs):
        self._backend = self.__SL
        self.checkpoints_(kwargs)
        self._backend = joblib
