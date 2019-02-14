#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.stats import boxcox
from sklearn.base import BaseEstimator
from sklearn.preprocessing import minmax_scale


class DescriptorHeatmap(BaseEstimator):
    """
    Heatmap.
    """

    def __init__(self,
                 save=None,
                 bc=False,
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

    def _transform(self, series):
        series_ = series
        if series.min() != series.max():
            if self.bc:
                with np.errstate(all='raise'):
                    shift = 1e-10
                    tmp = series - series.min() + shift
                    try:
                        series_, _ = boxcox(tmp)
                    except FloatingPointError:
                        series_ = series
        series_ = minmax_scale(series_)
        return series_

    def fit(self, desc):
        desc_ = desc.apply(self._transform)
        self.desc = pd.DataFrame(desc_, index=desc.index, columns=desc.columns)
        return self

    def draw(self, y=None):
        ax = sb.clustermap(
            self.desc,
            cmap="RdBu",
            method=self.method,
            figsize=self.figsize,
            row_cluster=self.row_cluster,
            col_cluster=self.col_cluster,
            **self.kwargs)
        ax.cax.set_visible(False)
        ax.ax_heatmap.yaxis.set_ticks_position('left')
        ax.ax_heatmap.yaxis.set_label_position('left')

        if y is None:
            ax.ax_col_dendrogram.set_position((0.1, 0.8, 0.9, 0.1))
            ax.ax_heatmap.set_position((0.1, 0.2, 0.9, 0.6))
        else:
            ax.ax_col_dendrogram.set_position((0.1, 0.8, 0.83, 0.1))
            ax.ax_heatmap.set_position((0.1, 0.2, 0.84, 0.6))
            ax = plt.axes([0.95, 0.2, 0.05, 0.6])
            x_ = y.values
            y_ = np.arange(len(x_))[::-1]
            ax.plot(x_, y_, lw=4)
            ax.get_yaxis().set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xlabel('{:s}'.format(y.name), fontsize='large')
        if self.save:
            plt.savefig(**self.save)
