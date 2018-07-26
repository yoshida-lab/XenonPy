# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split


class Splitter(object):
    """
    Data splitter for train and test
    """

    def __init__(self,
                 total_size,
                 *,
                 test_size=0.2,
                 random_state=None,
                 shuffle=True):
        """
        Parameters
        ----------
        total_size: int
            Total sample size.
        test_size: float, int, None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. By default, the value is set to 0.2.
        random_state: int, RandomState instance or None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`. Default is None.
        shuffle: boolean
            Whether or not to shuffle the data before splitting.
        """
        self._size = np.arange(total_size)
        self._random_state = random_state
        self._shuffle = shuffle
        self._test_size = test_size
        self._test = None
        self._train = None
        self.re_sample(
            test_size=test_size, random_state=random_state, shuffle=shuffle)

    def re_sample(self, *, test_size=0.2, random_state=None, shuffle=True):
        """
        Re-sample date.

        Parameters
        ----------
        test_size: float, int, None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. By default, the value is set to 0.2.
        random_state: int, RandomState instance or None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`. Default is None.
        shuffle: boolean
            Whether or not to shuffle the data before splitting.
        """
        self._random_state = random_state or self._random_state
        self._test_size = test_size or self._test_size
        self._shuffle = shuffle or self._shuffle
        self._train, self._test = train_test_split(
            self._size,
            test_size=self._test_size,
            random_state=self._random_state,
            shuffle=self._shuffle)

    @property
    def index(self):
        """
        Return training and testing data index.

        Returns
        -------
        tuple
            A tuple as (train_index, test_index).
        """
        return self._train, self._test

    @property
    def size(self):
        return self._size.size

    def split(self, *arrays, train=True, test=True):
        """
        Split data with index.

        Parameters
        ----------
        *arrays: DataFrame, Series, ndarray or list
            Sequence of indexables with same length / shape[0] as array that used to init class.
        train: bool
            Return training data if true.
        test: bool
            Return test data if true.

        Returns
        -------
        tuple
            List containing train-test split of inputs.
            length=2 * len(arrays)
        """
        if not (train or test):
            raise ValueError("train and test can't simultaneously False ")

        def _split(array):
            ret = []
            if isinstance(array, (DataFrame, Series)):
                if array.shape[0] != self._size.size:
                    raise ValueError(
                        'para `data` must have row size {} but got {}'.format(
                            self._size.size, array.shape[0]))
                if train:
                    ret.append(array.iloc[self._train])
                if test:
                    ret.append(array.iloc[self._test])
                return ret
            if isinstance(array, list):
                array = np.array(array)
            if isinstance(array, np.ndarray):
                if array.shape[0] != self._size.size:
                    raise ValueError(
                        'para `data` must have row size {} but got {}'.format(
                            self._size.size, array.shape[0]))
                if train:
                    ret.append(array[self._train])
                if test:
                    ret.append(array[self._test])
                return ret
            raise TypeError(
                'must be a list or matrix like object but got {}.'.format(
                    array))

        if len(arrays) == 1:
            return _split(arrays[0])
        ret_ = [_split(array) for array in arrays]
        ret__ = []
        for s in ret_:
            ret__ += s
        return tuple(ret__)
