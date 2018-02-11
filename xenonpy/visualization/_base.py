# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from copy import deepcopy

from numpy import ndarray
from pandas import Series
from plotly.graph_objs import Figure


class BasePlot(object):
    """
    Base plot.
    """
    _config = {'displaylogo': False}
    _marker = dict(size=6, opacity=0.8)  # todo: add with style(style_name) in future
    _layout = dict(
        xaxis={
            'linecolor': 'black',
            'linewidth': 2,
            'mirror': True,
            'showgrid': True,
            'showline': True,
            'zeroline': False
        },
        yaxis={
            'linecolor': 'black',
            'linewidth': 2,
            'mirror': True,
            'showgrid': True,
            'showline': True,
            'zeroline': False
        },
        autosize=False,
        showlegend=True,
        width=1000,
        height=1000
    )

    def __init__(self, trace):
        self.__traces = [trace]

    @property
    def trace(self):
        raise NotImplementedError()

    @property
    def layout(self):
        return self._layout

    def config(self, cfg):
        self._config = dict(self._config, **cfg)

    def show(self, filename=None, offline=True):
        if offline:
            from plotly.offline import plot
        else:
            from plotly.plotly import plot
        plot(self.figure, filename=filename, config=self._config)

    def ishow(self, filename=None, offline=True, connected=True):
        if offline:
            from plotly.offline import iplot, init_notebook_mode
            init_notebook_mode(connected=connected)
            iplot(self.figure, filename=filename, config=self._config)

        else:
            from plotly.plotly import iplot
            iplot(self.figure, filename=filename, config=self._config)

    @property
    def figure(self):
        if len(self.__traces) > 1:
            for i, trace in enumerate(self.__traces):
                if 'unnamed' in trace['name']:
                    trace['name'] = 'unnamed' + str(i + 1)

        return Figure(data=self.__traces, layout=self._layout)

    def __mul__(self, other):
        if not isinstance(other, BasePlot):
            raise ValueError('only object inherit from BasePlot can use add')

        cp = deepcopy(self)
        cp.__traces.append(deepcopy(other.trace))
        cp._layout = dict(cp.layout, **other.layout)

        return cp

    @staticmethod
    def check_data(x, y, data):
        def check(s):
            if isinstance(s, str) and data is not None:
                ret = data[s]
            elif isinstance(s, (list, Series, ndarray)):
                ret = s
            else:
                raise ValueError("'x' and 'y' must be array of numbers or a string ref to 'data'.")

            # validate x and y are filled with numbers only
            # if all(isinstance(element, Number) for element in ret):
            #     raise ValueError(
            #         "All elements of your 'x' and 'y' lists must be numbers but got {}.".format(type(ret[0])))

            return ret

        x = check(x)
        y = check(y)
        # validate x and y are the same length
        if len(x) != len(y):
            raise IndexError(
                "Both lists 'x' and 'y' must be the same length."
            )

        return x, y
