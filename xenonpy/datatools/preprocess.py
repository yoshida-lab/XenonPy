#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split, KFold


class Splitter(object):
    """
    Data splitter for train and test
    """

    def __init__(self,
                 sample_size,
                 *,
                 test_size=0.2,
                 cv=None,
                 random_state=None,
                 shuffle=True):
        """
        Parameters
        ----------
        sample_size: int
            Total sample size.
        test_size: float, int, None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. By default, the value is set to 0.2.
        cv: int, Series, list of string
            Number of folds. Must be at least 2. Default value is None.
        random_state: int, RandomState instance or None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`. Default is None.
        shuffle: boolean
            Whether or not to shuffle the data before splitting.
        """
        self._sample_size = np.arange(sample_size)
        self._cv = cv
        self._random_state = random_state
        self._shuffle = shuffle
        self._test_size = test_size
        self._test = None
        self._train = None
        self.roll(
            test_size=test_size, random_state=random_state, shuffle=shuffle)

    def roll(self, *, sample_size=None, test_size=0.2, cv=None, random_state=None, shuffle=True):
        """
        Generate the indices of train-test data.

        Parameters
        ----------
        sample_size: int
            The row size of samples.
        test_size: float, int, None
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. By default, the value is set to 0.2.
        cv: int or string
            Number of folds. Must be at least 2. Default value is None.
        random_state: int, RandomState instance or None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`. Default is None.
        shuffle: boolean
            Whether or not to shuffle the data before splitting.
        """
        self._random_state = random_state
        self._cv = cv if cv is not None else self._cv
        self._sample_size = sample_size if sample_size is not None else self._sample_size
        self._test_size = test_size if test_size is not None else self._test_size
        self._shuffle = shuffle if shuffle is not None else self._shuffle
        self._train, self._test = train_test_split(
            self._sample_size,
            test_size=self._test_size,
            random_state=self._random_state,
            shuffle=self._shuffle)
        return self

    @property
    def size(self):
        return self._sample_size.size

    @staticmethod
    def _split(array, train, validate=None, test=None):
        def _size_check(array_):
            size = train.size
            if validate is not None:
                size = size + validate.size
            if test is not None:
                size = size + test.size
            if array_.shape[0] != size:
                raise ValueError(
                    'para `data` must have row size {} but got {}'.format(size, array_.shape[0]))

        # all to np.array
        if isinstance(array, list):
            array = np.array(array)
        if isinstance(array, (DataFrame, Series)):
            array = array.values
        if not isinstance(array, np.ndarray):
            raise TypeError(
                'must be a list or matrix-like object but got {}.'.format(array))

        # check size
        _size_check(array)

        # split data
        ret = [array[train]]
        if validate is not None:
            ret.append(array[validate])
        if test is not None:
            ret.append(array[test])
        return ret

        # raise error

    def cv(self, *arrays, less_for_train=False):
        """
        Split data with cross-validation.

        Parameters
        ----------
        *arrays: DataFrame, Series, ndarray, list
            Data for split. Must be a Sequence of indexables with same length / shape[0].
            If None, return the split indices.
        less_for_train: bool
            If true, use less data set for train.
            E.g. ``[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]`` with 5 cv will be split into
            ``[1, 2]`` and ``[3, 4, 5, 6, 7, 8, 9, 0]``. Usually, ``[1, 2]`` (less one)
            will be used for validate. With ``less_for_train=True``, ``[1, 2]`` will be
            used for train. Default is ``False``.

        Yields
        -------
        list
            list containing split of inputs with cv. if inputs are None, only return
            the indices of split. if ``test_size`` is 0, test data/index will
            not return.
        """

        def group_cv():
            group = self._cv
            if isinstance(group, (list, np.ndarray)):
                group = Series(group)
            if not isinstance(group, Series):
                raise TypeError(
                    'must be a list or matrix-like object but got {}.'.format(group))
            group = group.rename('flag').reset_index(drop=True)
            group = group.iloc[self._train]

            for g in set(group):
                pure = group[group == g].index.values
                mixed = group[group != g].index.values
                if self._shuffle:
                    np.random.shuffle(pure)
                    np.random.shuffle(mixed)

                yield mixed, pure

            return

        if self._cv is None:
            raise ValueError('para `cv` must be set')

        if isinstance(self._cv, int):
            cv = KFold(n_splits=self._cv, shuffle=self._shuffle).split(self._train)
        else:
            cv = group_cv()
        for train, validate in cv:
            if less_for_train:
                train_, validate_, test_ = self._train[validate], self._train[train], self._test
            else:
                train_, validate_, test_ = self._train[train], self._train[validate], self._test
            if test_.size != 0:
                if len(arrays) == 0:
                    yield train_, validate_, test_
                else:
                    yield tuple([self._split(array, train_, validate_, test_) for array in arrays])
            else:
                if len(arrays) == 0:
                    yield train_, validate_
                else:
                    yield tuple([self._split(array, train_, validate_) for array in arrays])
        return

    def split(self, *arrays):
        """
        Split data.

        Parameters
        ----------
        *arrays: DataFrame, Series, ndarray, list
            Data for split. Must be a Sequence of indexables with same length / shape[0].
            If None, return the split indices.

        Returns
        -------
        tuple
            List containing split of inputs. if inputs are None, only return
            the indices of split. if ``test_size`` is 0, test data/index will
            not return.
        """
        if len(arrays) == 0:
            return self._train, self._test

        # if len(arrays) == 1:
        #     return _split(arrays[0])
        ret_ = [self._split(array, self._train, None, self._test) for array in arrays]
        ret__ = []
        for s in ret_:
            ret__ += s
        return tuple(ret__)
