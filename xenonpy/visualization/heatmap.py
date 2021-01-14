#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from typing import Sequence, Union

from xenonpy.datatools import Scaler

from sklearn.preprocessing import power_transform
from sklearn.base import BaseEstimator
from sklearn.preprocessing import minmax_scale


class DescriptorHeatmap(BaseEstimator):
    """
    Heatmap.
    """

    def __init__(self,
                 save=None,
                 bc=False,
                 cmap='RdBu',
                 pivot_kws=None,
                 method='average',
                 metric='euclidean',
                 figsize=None,
                 row_cluster=False,
                 col_cluster=True,
                 row_linkage=None,
                 col_linkage=None,
                 row_colors=None,
                 col_colors=None,
                 mask=None,
                 **kwargs):
        """

        Parameters
        ----------
        save
        bc
        pivot_kws
        method
        metric
        figsize
        row_cluster
        col_cluster
        row_linkage
        col_linkage
        row_colors
        col_colors
        mask
        kwargs
        """
        self.cmap = cmap
        self.save = save
        self.bc = bc
        self.pivot_kws = pivot_kws
        self.method = method
        self.metric = metric
        self.figsize = figsize
        self.col_cluster = col_cluster
        self.row_linkage = row_linkage
        self.row_cluster = row_cluster
        self.col_linkage = col_linkage
        self.row_colors = row_colors
        self.col_colors = col_colors
        self.mask = mask
        self.kwargs = kwargs
        self.desc = None

    def fit(self, desc):
        scaler = Scaler().min_max()
        if self.bc:
            scaler = scaler.yeo_johnson()

        self.desc = scaler.fit_transform(desc)
        return self

    def draw(self,
             y: Union[Sequence, pd.Series, None] = None,
             name: str = None,
             *,
             return_sorted_idx: bool = False):
        """
        Draw figure.

        Parameters
        ----------
        y
            Properties values corresponding to samples.
        name
            Property name. If ``name`` is ``None`` and ``y`` is not a ``pandas.Series``.
            No name will be draw in the figure.
        return_sorted_idx
            If ``True``, return sorted column index of descriptor.
            Default ``False``

        Returns
        -------
        idx: np.array
            sorted column index if ``return_sorted_idx`` is ``True``

        """
        heatmap_ax = sb.clustermap(self.desc,
                                   cmap=self.cmap,
                                   method=self.method,
                                   figsize=self.figsize,
                                   row_cluster=self.row_cluster,
                                   col_cluster=self.col_cluster,
                                   **self.kwargs)
        heatmap_ax.cax.set_visible(False)
        heatmap_ax.ax_heatmap.yaxis.set_ticks_position('left')
        heatmap_ax.ax_heatmap.yaxis.set_label_position('left')

        if y is None:
            heatmap_ax.ax_col_dendrogram.set_position((0.1, 0.8, 0.9, 0.1))
            heatmap_ax.ax_heatmap.set_position((0.1, 0.2, 0.9, 0.6))
        else:
            heatmap_ax.ax_col_dendrogram.set_position((0.1, 0.8, 0.83, 0.1))
            heatmap_ax.ax_heatmap.set_position((0.1, 0.2, 0.84, 0.6))
            prop_ax = plt.axes([0.95, 0.2, 0.05, 0.6])

            # draw prop
            if isinstance(y, pd.Series):
                x_ = y.values
                name_ = y.name
            else:
                x_ = np.asarray(y)
                name_ = ''
            if name is not None:
                name_ = name
            y_ = np.arange(len(x_))[::-1]

            prop_ax.plot(x_, y_, lw=4)
            prop_ax.get_yaxis().set_visible(False)
            prop_ax.spines['top'].set_visible(False)
            prop_ax.spines['right'].set_visible(False)
            prop_ax.set_xlabel('{:s}'.format(name_), fontsize='large')

        if self.save:
            plt.savefig(**self.save)

        if return_sorted_idx:
            try:
                return heatmap_ax.dendrogram_col.reordered_ind
            except AttributeError:
                return np.arange(heatmap_ax.data2d.shape[1])
