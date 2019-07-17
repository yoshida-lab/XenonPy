#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.
from functools import partial

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn


class Init(object):
    @staticmethod
    def uniform(*, scale=0.1):
        b = 1 * scale
        a = -b
        return partial(nn.init.uniform, a=a, b=b)


class L1(object):
    @staticmethod
    def conv(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.Conv1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.Conv1d
        """
        return partial(nn.Conv1d, *args, **kwargs)

    @staticmethod
    def linear(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.Linear`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.Linear
        """
        return partial(nn.Linear, *args, **kwargs)

    @staticmethod
    def batch_norm(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.BatchNorm1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.BatchNorm1d
        """
        return partial(nn.BatchNorm1d, *args, **kwargs)

    @staticmethod
    def instance_norm(*args, **kwargs):
        """
        Wrapper class for :class:`torch.nn.InstanceNorm1d`.
        http://pytorch.org/docs/0.3.0/optim.html#torch.nn.InstanceNorm1d
        """
        return partial(nn.InstanceNorm1d, *args, **kwargs)


class Metrics(object):
    @staticmethod
    def reg(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        pr, p_val = pearsonr(y_true, y_pred)
        return dict(
            mae=mae,
            rmse=rmse,
            r2=r2,
            pearsonr=pr,
            p_value=p_val
        )

    # @staticmethod
    # def cls(y_true, y_pred):
    #     mae = mean_absolute_error(y_true, y_pred)
    #     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    #     r2 = r2_score(y_true, y_pred)
    #     pr, p_val = pearsonr(y_true, y_pred)
    #     return dict(
    #         mae=mae,
    #         rmse=rmse,
    #         r2=r2,
    #         pearsonr=pr,
    #         p_value=p_val
    #     )
