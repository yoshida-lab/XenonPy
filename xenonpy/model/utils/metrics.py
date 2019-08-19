#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

__all__ = ['regression_metrics']


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> OrderedDict:
    if len(y_true.shape) != 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) != 1:
        y_pred = y_pred.flatten()

    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = mean_absolute_error(y_true, y_pred)
    maxae = max_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pr, p_val = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    return OrderedDict(
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        pearsonr=pr,
        spearmanr=sr,
        p_value=p_val,
        max_ae=maxae
    )
