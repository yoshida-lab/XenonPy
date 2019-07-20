#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from typing import Union, Tuple

from xenonpy.model.nn.training.base import BaseExtension
from xenonpy.model.nn.utils import T_Data, to_tensor, regression_metrics

__all__ = ['Validator']


class Validator(BaseExtension):

    def __init__(self,
                 x_test: Union[T_Data, Tuple[T_Data]],
                 y_test: T_Data,
                 *,
                 regression: bool = True
                 ):
        if not isinstance(x_test, tuple):
            self.x_test = (to_tensor(x_test),)
        else:
            self.x_test = [to_tensor(x_) for x_ in x_test]

        self.y_test = to_tensor(y_test).numpy().flatten()
        self.regression = regression

    def run(self, step_info, trainer):
        y_pred = trainer.predict(self.x_test).flatten()
        if self.regression:
            metrics = regression_metrics(y_pred, self.y_test)
            step_info.update(test_mae=metrics['mae'], test_rmse=metrics['rmse'], test_r2=metrics['r2'])
