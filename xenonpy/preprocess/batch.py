# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


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
