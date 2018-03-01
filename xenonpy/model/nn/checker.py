# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from pathlib import Path

import torch

from ..._conf import get_data_loc
from ...utils.datatools import Saver


class _SL(object):
    def __init__(self):
        self.load = torch.load
        self.dump = torch.save


class Checker(object):
    """
    Check point.
    """

    def __init__(self, name, path=None):
        """
        Parameters
        ----------
        name: str
            Model name.
        path: str
            Save path.
        """
        if path is None:
            path = get_data_loc('usermodel')

        i = 1
        while Path(path + '/' + name + '@' + str(i)).exists():
            i += 1
        _fpath = Path(path + '/' + name + '@' + str(i))
        self.name = _fpath.stem
        self.saver = Saver(self.name, path=path)
        self.saver.pkl = _SL()

    def __getitem__(self, item):
        if isinstance(item, int):
            return dict(model_state=self.saver['model_state', item],
                        epochs=self.saver['epochs', item],
                        y_pred=self.saver['y_pred', item],
                        loss=self.saver['loss', item])
        if isinstance(item, tuple):
            return self.saver.__getitem__(item)

        raise ValueError('except int or slice like [str, int]')

    def __call__(self, **kwargs):
        self.saver(**kwargs)
