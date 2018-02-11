# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


from pandas import DataFrame
from plotly.graph_objs import Scattergl

from ._base import BasePlot


class RegPlot(BasePlot):
    """
    Plot data and a linear regression model fit.
    """

    @property
    def trace(self):
        return self._trace

    def __init__(self, x, y, *,
                 data=None,
                 label='unnamed',
                 marker=None,
                 layout=None):
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

        self._trace = Scattergl(dict(
            x=self._x,
            y=self._y,
            mode='markers',
            name=label,
            marker=self._marker)
        )
        super().__init__(self._trace)
