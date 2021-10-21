#  Copyright (c) 2021. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

__all__ = ['regression_metrics', 'classification_metrics']


def regression_metrics(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> OrderedDict:
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
    y_pred: Union[np.ndarray, pd.Series],
    *,
    average: Union[None, List[str], Tuple[str]] = ('weighted', 'micro', 'macro'),
    labels=None,
) -> dict:
    """
    Calculate most common classification scores.
    See also: https://scikit-learn.org/stable/modules/model_evaluation.html
    
    Parameters
    ----------
    y_true
        True results.
    y_pred
        Predicted results.
    average
        This parameter is required for multiclass/multilabel targets. If None, the scores for each class are returned.
        Otherwise, this determines the type of averaging performed on the data:

        binary:
            Only report results for the class specified by pos_label. This is applicable only if targets (y_{true,pred})
            are binary.
        micro:
            Calculate metrics globally by counting the total true positives, false negatives and false positives.
        macro:
            Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into
            account.
        weighted:
            Calculate metrics for each label, and find their average weighted by support (the number of true instances
            for each label). This alters ``macro`` to account for label imbalance; it can result in an F-score that is
            not between precision and recall.
    labels
        The set of labels to include when average != ``binary``, and their order if average is None.
        Labels present in the data can be excluded, for example to calculate a multiclass average ignoring a majority
        negative class, while labels not present in the data will result in 0 components in a macro average.
        For multilabel targets, labels are column indices.
        By default, all labels in y_true and y_pred are used in sorted order.

    Returns
    -------
    OrderedDict
        An :class:`collections.OrderedDict` contains classification scores.
        These scores will always contains ``accuracy``, ``f1``, ``precision`` and ``recall``.
        For multilabel targets, based on the selection of the ``average`` parameter, the **weighted**, **micro**,
        and **macro** scores of ``f1`, ``precision``, and ``recall`` will be calculated.
    """
    if average is not None and len(average) == 0:
        raise ValueError('need average')

    if len(y_true.shape) != 1:
        y_true = np.argmax(y_true, 1)
    if len(y_pred.shape) != 1:
        y_pred = np.argmax(y_pred, 1)

    mask = ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    ret = dict(accuracy=accuracy_score(y_true, y_pred))

    ret.update(
        f1=f1_score(y_true, y_pred, average=None, labels=labels),
        precision=precision_score(y_true, y_pred, average=None, labels=labels),
        recall=recall_score(y_true, y_pred, average=None, labels=labels),
    )

    if 'binary' in average:
        ret.update(
            binary_f1=f1_score(y_true, y_pred, average='binary', labels=labels),
            binary_precision=precision_score(y_true, y_pred, average='binary', labels=labels),
            binary_recall=recall_score(y_true, y_pred, average='binary', labels=labels),
        )

    if 'micro' in average:
        ret.update(
            micro_f1=f1_score(y_true, y_pred, average='micro', labels=labels),
            micro_precision=precision_score(y_true, y_pred, average='micro', labels=labels),
            micro_recall=recall_score(y_true, y_pred, average='micro', labels=labels),
        )

    if 'macro' in average:
        ret.update(
            macro_f1=f1_score(y_true, y_pred, average='macro', labels=labels),
            macro_precision=precision_score(y_true, y_pred, average='macro', labels=labels),
            macro_recall=recall_score(y_true, y_pred, average='macro', labels=labels),
        )
        
    if 'weighted' in average:
        ret.update(
            weighted_f1=f1_score(y_true, y_pred, average='weighted', labels=labels),
            weighted_precision=precision_score(y_true, y_pred, average='weighted', labels=labels),
            weighted_recall=recall_score(y_true, y_pred, average='weighted', labels=labels),
        )

    if 'samples' in average:
        ret.update(
            samples_f1=f1_score(y_true, y_pred, average='samples', labels=labels),
            samples_precision=precision_score(y_true, y_pred, average='samples', labels=labels),
            samples_recall=recall_score(y_true, y_pred, average='samples', labels=labels),
        )

    return ret
