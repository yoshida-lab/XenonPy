# Copyright 2021 TsumiNa
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
from typing import Union, Sequence
from sklearn.preprocessing import MinMaxScaler
from xenonpy.datatools import preset

__all__ = ['rbf_kernel' 'calculate_rbf_kernel_matrix']


def rbf_kernel(x_i: np.ndarray, x_j: Union[np.ndarray, int, float], sigmas: Union[float, int, np.ndarray,
                                                                                  Sequence]) -> np.ndarray:
    """
    Radial Basis Function (RBF) kernel function.
    https://en.wikipedia.org/wiki/Radial_basis_function_kernel

    Parameters
    ----------
    sigmas:
        The standard deviations (SD).
        Can be a single number or a 1d array-like object.
    x_i:
        Should be a 1d array.
    x_j : np.ndarray
        Should be a 1d array.

    Returns
    -------
    np.ndarray
        Distribution under RBF kernel.

    Raises
    ------
    ValueError
        Raise error if sigmas has wrong dimension.
    """
    sigmas = np.asarray(sigmas)
    if sigmas.ndim == 0:
        sigmas = sigmas[np.newaxis]
    if sigmas.ndim != 1:
        raise ValueError('parameter `sigmas` must be a array-like object which has dimension 1')

    # K(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 * sigma^2))
    p1 = np.power(np.expand_dims(x_i, axis=x_i.ndim) - x_j, 2)
    p2 = np.power(sigmas, 2) * 2
    dists = np.exp(-np.expand_dims(p1, axis=p1.ndim) / p2).transpose([2, 0, 1])

    if dists.shape[0] == 1:
        return dists[0]
    return dists


def calculate_rbf_kernel_matrix(
        *,
        element_info: Union[None, pd.DataFrame] = None,
        scaled_element_info: bool = False,
        quartiles: Sequence[int] = (25, 50, 75),
        half_interval_by_sigma: float = 2,
        sort_centers: bool = True,
):
    if element_info is None:
        element_info = preset.elements_completed

    if scaled_element_info:
        element_info = pd.DataFrame(MinMaxScaler().fit_transform(element_info),
                                    columns=element_info.columns,
                                    index=element_info.index)

    all_dists = []
    center_labels = []
    for feature, data in element_info.iteritems():
        if sort_centers:
            data = data.values
            centers = np.unique(data)
        else:
            centers = data.unique()
            data = data.values
        intervals = np.unique([abs(i - j) for i, j in zip(data[:-1], data[1:])])  # get all intervals
        quartiles = np.percentile(intervals / 2, [25, 50, 75])  # get 25%, 50%, 75% quantile of intervals / 2
        sigmas = quartiles / half_interval_by_sigma  # use unique quantiles as sigma of RBF kernel

        # RBF kernel
        dists = rbf_kernel(data, centers, sigmas)
        all_dists.append(dists)
        center_labels.append(pd.Series(centers, index=[feature] * centers.size))

    return all_dists, pd.concat(center_labels)
