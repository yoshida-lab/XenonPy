# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


from pandas import DataFrame
from plotly.graph_objs import Scattergl

from .._base import BasePlot


class Scatter(BasePlot):
    """
    Plot data and a linear regression model fit.
    """

    @property
    def annotation(self):
        return self._annotation

    @property
    def trace(self):
        return self._trace

    def __init__(self, x, y, *,
                 data=None,
                 label=None,
                 marker=None,
                 layout=None,
                 annotation=None):
        """"""
        if data:
            if not isinstance(data, DataFrame):
                raise ValueError("'data' must be a 'DataFrame' object.")
            self._data = data
        if marker:
            self._marker = dict(self._marker, **marker)
        if layout:
            self._layout = dict(self._layout, **layout)
        self._x, self._y = self.check_data(x, y, data)
        self._annotation = annotation

        _trace = dict(
            x=self._x,
            y=self._y,
            mode='markers',
            marker=self._marker)
        if label:
            _trace['name'] = label

        self._trace = Scattergl(_trace)
        super().__init__(self._trace)
