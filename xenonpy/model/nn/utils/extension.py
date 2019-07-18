#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from collections import OrderedDict
from typing import Union, Tuple

import torch
from torch.utils.data import DataLoader

from xenonpy.model.nn.utils.base import BaseExtension


class Validator(BaseExtension):

    def __init__(self,
                 x_test: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
                 y_test: torch.Tensor = None,
                 *,
                 test_dataset: DataLoader = None,
                 ):
        if test_dataset is not None:
            if x_test is not None or y_test is not None:
                raise RuntimeError('parameter <data_loader> is exclusive of <x_train> and <y_train>')
        else:
            if y_test is None or x_test is None:
                raise RuntimeError('missing parameter <x_train> or <y_train>')

    def run(self, step_info: OrderedDict, trainer):
        if x_test is not None and y_test is not None:
            y_pred, y = self.predict(x_test), y_test.to(self._device)
            step_info['test_loss'] = self.loss_func(y_pred, y).item()
            self._model.train()
