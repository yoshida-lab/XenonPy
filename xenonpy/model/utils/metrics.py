#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

__all__ = ['regression_metrics', 'classification_metrics']


def regression_metrics(y_true: Union[np.ndarray, pd.Series],
                       y_pred: Union[np.ndarray, pd.Series]) -> OrderedDict:
    """
    Calculate most common regression scores.
    See Also: https://scikit-learn.org/stable/modules/model_evaluation.html
    
    Parameters
    ----------
    y_true
        True results.
    y_pred
        Predicted results.
        
    Returns
    -------
    OrderedDict
        An :class:`collections.OrderedDict` contains regression scores.
        These scores will be calculated: ``mae``, ``mse``, ``rmse``, ``r2``,
        ``pearsonr``, ``spearmanr``, ``p_value``, and ``max_ae``
    """
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
        max_ae=maxae,
    )


def classification_metrics(
        y_true: Union[np.ndarray, pd.DataFrame, pd.Series],
        y_pred: Union[np.ndarray, pd.Series]) -> OrderedDict:
    """
    Calculate most common classification scores.
    See Also: https://scikit-learn.org/stable/modules/model_evaluation.html
    
    Parameters
    ----------
    y_true
        True results.
    y_pred
        Predicted results.
        
    Returns
    -------
    OrderedDict
        An :class:`collections.OrderedDict` contains classification scores.
        These scores will be calculated: ``accuracy``, ``f1``, ``precision``, ``recall``,
        ``macro_f1``, ``macro_precision``, and ``macro_recall``
    """
    if len(y_true.shape) != 1:
        y_true = np.argmax(y_true, 1)
    if len(y_pred.shape) != 1:
        y_pred = np.argmax(y_pred, 1)

    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    return OrderedDict(
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
        macro_f1=macro_f1,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
    )
