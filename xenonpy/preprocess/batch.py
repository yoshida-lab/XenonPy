# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .transform import BoxCox


class _Valuer(object):
    def __init__(self, value):
        self.__value = value
        self.__now = value
        self._test = None
        self.__inverse_chain = []

    @property
    def box_cox(self):
        scaler = BoxCox()
        self.__now = scaler.fit_transform(self.__now)
        self.__inverse_chain.append(scaler.inverse_transform)
        return self

    @property
    def min_max(self):
        scaler = MinMaxScaler()
        self.__now = scaler.fit_transform(self.__now)
        self.__inverse_chain.append(scaler.inverse_transform)
        return self

    @property
    def standard_scale(self):
        scaler = StandardScaler()
        self.__now = scaler.fit_transform(self.__now)
        self.__inverse_chain.append(scaler.inverse_transform)
        return self

    @property
    def value(self):
        return self.__now

    @property
    def test(self):
        return self.__now.loc[self._test, :]

    # fixme: from here
    def inverse(self, data):
        for inv in self.__inverse_chain[::-1]:
            data = inv(data)
        return data

    def drop_transform(self):
        self.__now = self.__value


class Batch(object):
    """
    Batch data for ML
    """

    def __init__(self, x, y=None, describe=None):
        self._x = x
        self._y = y
        self._describe = describe

    def sort_property(self, ascend=False):
        raise NotImplementedError()

    def split(self, test_rate=0.2):
        raise NotImplementedError()
