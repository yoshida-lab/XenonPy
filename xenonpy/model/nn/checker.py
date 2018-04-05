# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from pathlib import Path
from platform import version, system

import numpy
import pandas
import torch
from sklearn.externals import joblib

from ... import __version__
from ...utils import get_data_loc, DataSet


class _SL(object):
    load = torch.load
    dump = torch.save


class Checker(DataSet):
    """
    Check point.
    """

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
    def load(cls, name, path=None):
        return cls(name, path, increment=False)

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

    @describe.setter
    def describe(self, description):
        """
        Set description.

        Parameters
        ----------
        description: dict
            Description in dict object.
        """
        desc = dict(
            python=version(),
            system=system(),
            numpy=numpy.__version__,
            torch=torch.__version__,
            xenonpy=__version__)
        if isinstance(description, dict):
            desc = dict(desc, **description)
        else:
            raise TypeError('except dict but got {}'.format(type(description)))
        super().__call__(describe=desc)

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
        self._backend = _SL
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
            self._backend = _SL
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
        self._backend = _SL
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
            self._backend = _SL
            super().__call__(trained_model=model)
            self._backend = joblib
        else:
            raise TypeError(
                'except `torch.nn.Module` object but got {}'.format(
                    type(model)))

    def train_data(self, x_train, y_train, x_id=None, y_id=None):
        """
        Save training data.

        Parameters
        ----------
        x_train: pandas.DataFrame or numpy.ndarray
            Features as X.
        y_train: pandas.DataFrame or numpy.ndarray
            Target property as y.
        x_id: list-like
            Row id for train
        y_id: list-like
            Row id for test
        """

        def _check(o):
            return isinstance(o, (numpy.ndarray, pandas.DataFrame))

        if _check(x_train) and _check(y_train):
            super().__call__(x_train=x_train, y_train=y_train)
            if x_id:
                super().__call__(x_id=x_id)
            if y_id:
                super().__call__(y_id=y_id)
            return
        raise TypeError('except `numpy.ndarray or pandas.DataFrame`')

    def add_predict(self, **pred):
        """
        add a prediction result.
        This action don't overwrite but add a new object.

        Parameters
        ----------
        pred: dict
            A prediction result as dict.
        """
        self.predicts(**pred)

    def __getitem__(self, item):
        if isinstance(item, int):
            self._backend = _SL
            try:
                return self.checkpoints[item]
            except IndexError:
                return None
            finally:
                self._backend = joblib
        raise TypeError('except int as checkpoint index but got {}'.format(
            type(item)))

    def __call__(self, **kwargs):
        self._backend = _SL
        self.checkpoints(kwargs)
        self._backend = joblib
