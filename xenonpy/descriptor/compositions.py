#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np

from .base import BaseDescriptor, BaseCompositionFeaturizer


class Counting(BaseCompositionFeaturizer):
    def __init__(self, *, one_hot_vec=False, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        one_hot_vec : bool
            Set ``true`` to using one-hot-vector encoding.
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.one_hot_vec = one_hot_vec
        self._elems = self._elements.index.tolist()

    def mix_function(self, elems, nums):
        vec = np.zeros(len(self._elems), dtype=np.int)
        for i, e in enumerate(elems):
            if self.one_hot_vec:
                vec[self._elems.index(e)] = 1
            else:
                vec[self._elems.index(e)] = nums[i]

        return vec

    @property
    def feature_labels(self):
        return self._elems


class WeightedAverage(BaseCompositionFeaturizer):
    def __init__(self, *, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values
        w_ = nums / np.sum(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['ave:' + s for s in self._elements]


class WeightedSum(BaseCompositionFeaturizer):
    def __init__(self, *, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values
        w_ = np.array(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['sum:' + s for s in self._elements]


class GeometricMean(BaseCompositionFeaturizer):
    def __init__(self, *, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values
        w_ = np.array(nums).reshape(-1, 1)
        tmp = elems_ ** w_
        return np.power(tmp.prod(axis=0), 1 / sum(w_))

    @property
    def feature_labels(self):
        return ['gmean:' + s for s in self._elements]


class HarmonicMean(BaseCompositionFeaturizer):
    def __init__(self, *, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = 1 / self._elements.loc[elems, :].values
        w_ = np.array(nums)
        tmp = w_.dot(elems_)

        return sum(w_) / tmp

    @property
    def feature_labels(self):
        return ['hmean:' + s for s in self._elements]


class WeightedVariance(BaseCompositionFeaturizer):
    def __init__(self, *, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, nums):
        elems_ = self._elements.loc[elems, :].values
        w_ = nums / np.sum(nums)
        mean_ = w_.dot(elems_)
        var_ = elems_ - mean_
        return w_.dot(var_ ** 2)

    @property
    def feature_labels(self):
        return ['var:' + s for s in self._elements]


class MaxPooling(BaseCompositionFeaturizer):
    def __init__(self, *, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, _):
        elems_ = self._elements.loc[elems, :]
        return elems_.max().values

    @property
    def feature_labels(self):
        return ['max:' + s for s in self._elements]


class MinPooling(BaseCompositionFeaturizer):
    def __init__(self, *, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def mix_function(self, elems, _):
        elems_ = self._elements.loc[elems, :]
        return elems_.min().values

    @property
    def feature_labels(self):
        return ['min:' + s for s in self._elements]


class Compositions(BaseDescriptor):
    """
    Calculate elemental descriptors from compound's composition.
    """

    def __init__(self, *, n_jobs=-1, featurizers='all', on_errors='nan'):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        featurizers: list[str] or 'all'
            Featurizers that will be used.
            Default is 'all'.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'nan' which will raise up the exception.
        """

        super().__init__(featurizers=featurizers)
        self.n_jobs = n_jobs

        self.composition = Counting(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = WeightedAverage(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = WeightedSum(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = WeightedVariance(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = GeometricMean(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = HarmonicMean(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = MaxPooling(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = MinPooling(n_jobs=n_jobs, on_errors=on_errors)
