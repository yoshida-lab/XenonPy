# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import pandas as pd
import numpy as np

from sklearn.preprocessing import minmax_scale
from sklearn.base import BaseEstimator
import seaborn as sb
import matplotlib.pyplot as plt
from ..decorator import combinator
from .. import Path


@combinator
class DescHeatmap(BaseEstimator):
    def __init__(self,
                 save=None,
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
        self.save = save
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
        self.desc = pd.DataFrame(
            minmax_scale(desc), index=desc.index, columns=desc.columns)
        return self

    def draw(self, y):
        ax = sb.clustermap(
            self.desc,
            cmap="RdBu",
            method=self.method,
            figsize=self.figsize,
            row_cluster=self.row_cluster,
            col_cluster=self.col_cluster,
            **self.kwargs)
        ax.cax.set_visible(False)
        ax.ax_heatmap.set_position((0.1, 0.2, 0.84, 0.6))
        ax.ax_heatmap.yaxis.set_ticks_position('left')
        ax.ax_heatmap.yaxis.set_label_position('left')
        ax.ax_col_dendrogram.set_position((0.1, 0.8, 0.83, 0.1))

        ax = plt.axes([0.95, 0.2, 0.05, 0.6])
        ax.plot(y.values, lw=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('{:s}\n(property)'.format(y.name), fontsize=20)
        if self.save:
            plt.savefig(**self.save)
