#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from functools import partial
from pathlib import Path

import torch
from sklearn.externals import joblib

from ...datatools.storage import Storage
from ...utils import get_data_loc


class Checker(Storage):
    """
    Check point.
    """

    class __SL(object):
        load = partial(torch.load, map_location=torch.device('cpu'))
        dump = torch.save

    def __init__(self, name, path=None, *, increment=True):
        """
        Parameters
        ----------
        name: str
            Model name.
        path: str
            Save path.
        """
        if not name:
            raise ValueError('need model name.')
        if path is None:
            path = get_data_loc('usermodel')

        if increment:
            i = 1
            while Path(path + '/' + name + '@' + str(i)).exists():
                i += 1
            _fpath = Path(path + '/' + name + '@' + str(i))
        else:
            _fpath = Path(path) / name
        self._name = _fpath.stem
        super().__init__(self._name, path=path)

    @classmethod
    def load(cls, model_path):
        p = Path(model_path)
        return cls(p.stem, str(p.parent), increment=False)

    @property
    def model_name(self):
        return self._name

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
                cp = self.checkpoints[item]
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
        self.checkpoints(kwargs)
        self._backend = joblib
