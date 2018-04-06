# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


class DataSplitter(object):
    """
    Batch data for ML
    """

    def __init__(self, data):
        if not isinstance(data, (DataFrame, Series, np.ndarray, list)):
            raise TypeError('must be a list or matrix like object but got {}.'.format(data))
        if isinstance(data, list):
            data = np.array(data)
        self._index = np.arange(data.shape[0])
        self._train_index = None
        self._test_index = None

    def sort_property(self, ascend=False):
        raise NotImplementedError()

    def split(self, test_size=0.2):  # fixme: from here
        self._train_index, self._test_index = train_test_split(self._index, test_size=test_size)

    @property
    def index(self):
        if self._train_index is None or self._test_index is None:
            self.split()
        return self._train_index, self._test_index
