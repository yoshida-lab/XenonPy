#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union, Tuple, Iterable, List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn import utils
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, KFold

__all__ = ['Splitter']


class Splitter(BaseEstimator):
    """
    Data splitter for train and test
    """

    def __init__(self,
                 size: int,
                 *,
                 test_size: Union[float, int] = 0.2,
                 k_fold: Union[int, Iterable, None] = None,
                 random_state: Union[int, None] = None,
                 shuffle: bool = True):
        """
        Parameters
        ----------
        size
            Total sample size.
            All data must have same length of their first dim,
        test_size
            If float, should be between ``0.0`` and ``1.0`` and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. Can be ``0`` if cv is ``None``.
            In this case, :meth:`~Splitter.cv` will yield a tuple only contains ``training`` and ``validation``
            on each step. By default, the value is set to 0.2.
        k_fold
            Number of k-folds.
            If ``int``, Must be at least 2.
            If ``Iterable``, it should provide label for each element which will be used for group cv.
            In this case, the input of :meth:`~Splitter.cv` must be a :class:`pandas.DataFrame` object.
            Default value is None to specify no cv.
        random_state
            If int, random_state is the seed used by the random number generator;
            Default is None.
        shuffle
            Whether or not to shuffle the data before splitting.
        """
        if k_fold is None and test_size == 0:
            raise RuntimeError('<test_size> can be zero only if <cv> is not none')
        self._k_fold = k_fold
        self._shuffle = shuffle
        self._test_size = test_size

        self._sample_size = np.arange(size)
        self._test: Union[np.ndarray, None] = None
        self._train: Union[np.ndarray, None] = None
        self._cv_indices: List[Tuple[np.ndarray, np.ndarray]] = []
        self._random_state = random_state
        self.roll(random_state)

    @property
    def size(self):
        return self._sample_size.size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def test_size(self):
        return self._test_size

    @property
    def k_fold(self):
        return self._k_fold

    @property
    def random_state(self):
        return self._random_state

    def roll(self, random_state: int = None):

        if self._test_size == 0:
            if self._shuffle:
                self._train = utils.shuffle(self._sample_size)
            else:
                self._train = self._sample_size
        else:
            self._train, self._test = train_test_split(self._sample_size,
                                                       test_size=self._test_size,
                                                       random_state=random_state,
                                                       shuffle=self._shuffle)

        if isinstance(self._k_fold, int):
            cv = KFold(n_splits=self._k_fold, shuffle=self._shuffle, random_state=random_state)
            for train, val in cv.split(self._train):
                self._cv_indices.append((self._train[train], self._train[val]))
        elif isinstance(self._k_fold, Iterable):
            tmp: pd.Series = pd.Series(self._k_fold).reset_index(drop=True).iloc[self._train]
            for g in set(tmp):
                val = tmp[tmp == g].index.values
                train = tmp[tmp != g].index.values
                self._cv_indices.append((train, val))

    def _check_input(self, array):
        if isinstance(array, (list, tuple)):
            array = np.asarray(array)
        if not isinstance(array, (np.ndarray, pd.DataFrame, pd.Series)):
            raise TypeError(
                f'<arrays> must be list, numpy.ndarray, pandas.DataFrame, or pandas.Series but got {array.__class__}.'
            )
        if array.shape[0] != self.size:
            raise ValueError(
                f'parameters <arrays> must have size {self.size} for dim 0 but got {array.shape[0]}'
            )
        return array

    @staticmethod
    def _split(array, *idx):

        # all to np.array
        if isinstance(array, np.ndarray):
            return [array[i] for i in idx]

        if isinstance(array, (DataFrame, Series)):
            return [array.iloc[i] for i in idx]

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
            will be used for val. With ``less_for_train=True``, ``[1, 2]`` will be
            used for train. Default is ``False``.

        Yields
        -------
        tuple
            list containing split of inputs with cv. if inputs are None, only return
            the indices of split. if ``test_size`` is 0, test data/index will
            not return.
        """

        if self._k_fold is None:
            raise RuntimeError('parameter <cv> must be set')

        for train, val in self._cv_indices:
            if less_for_train:
                tmp = train
                train = val
                val = tmp

            if len(arrays) == 0:
                if self._test is not None:
                    yield train, val, self._test
                else:
                    yield train, val
            else:
                ret = []
                for array in arrays:
                    array = self._check_input(array)
                    if self._test is not None:
                        ret.extend(self._split(array, train, val, self._test))
                    else:
                        ret.extend(self._split(array, train, val))
                yield tuple(ret)
        return

    def split(self, *arrays: Union[np.ndarray, pd.DataFrame, pd.Series]):
        """
        Split data.

        Parameters
        ----------
        *arrays
            Dataset for split.
            Size of dim 0 must be equal to :meth:`~Splitter.size`.
            If None, return the split indices.

        Returns
        -------
        tuple
            List containing split of inputs. if inputs are None, only return
            the indices of splits. if ``test_size`` is 0, test data/index will
            not return.
        """
        if self._test is None:
            raise RuntimeError('split action is illegal because `test_size` is none')

        if len(arrays) == 0:
            return self._train, self._test

        ret = []
        for array in arrays:
            array = self._check_input(array)
            ret.extend(self._split(array, self._train, self._test))
        return tuple(ret)
