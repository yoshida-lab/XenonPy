# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import unittest
from XenonPy import load


class TestDatasets(unittest.TestCase):
    '''Test dataset module'''

    @classmethod
    def setUpClass(cls):
        print('ignore NumPy RuntimeWarning\n')
        import warnings
        warnings.filterwarnings("ignore", message="numpy.dtype size changed")
        warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    def test_load(self):
        '''Test load function'''
        e = load('elements')
        self.assertEqual(118, e.shape[0])
        self.assertEqual(74, e.shape[1])
        e = load('mp_inorganic')
        self.assertEqual(69640, e.shape[0])
        self.assertEqual(15, e.shape[1])
        e = load('electron_density')
        self.assertEqual(103, e.shape[0])
        self.assertEqual(401, e.shape[1])