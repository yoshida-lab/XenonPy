#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
from pymatgen.core.composition import Composition as PMGComp

from .base import BaseFeaturizer, BaseDescriptor
from ..datatools.preset import preset


class _CompositionalFeature(BaseFeaturizer):

    def __init__(self, elements=None, *, n_jobs=-1, include=None,
                 exclude=None, on_errors='raise', return_type='any'):
        """
        Base class for composition feature.
        """

        if include and exclude:
            raise ValueError(
                'Paratemer "include" and "exclude" are mutually exclusive.')
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

        if elements is None:
            elements = preset.elements_completed
        if include is not None:
            elements = elements[include]
        if exclude is not None:
            elements = elements.drop(exclude, axis=1)

        self.elements = elements
        self.__authors__ = ['TsumiNa']

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, PMGComp):
            comp = comp.as_dict()
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        return self._func(elems_, nums_)

    def _func(self, elems, nums):
        raise NotImplementedError

    @property
    def feature_labels(self):
        """
        Generate attribute names.
        Returns:
            ([str]) attribute labels.
        """
        raise NotImplementedError("feature_labels() is not defined!")


class WeightedAvgFeature(_CompositionalFeature):
    def __init__(self, elements=None, *, n_jobs=-1, include=None,
                 exclude=None, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
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

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude, on_errors=on_errors,
            return_type=return_type)

    def _func(self, elems, nums):
        elems_ = self.elements.loc[elems, :]
        w_ = nums / np.sum(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['ave:' + s for s in self.elements]


class WeightedSumFeature(_CompositionalFeature):
    def __init__(self, elements=None, *, n_jobs=-1, include=None,
                 exclude=None, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
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

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude, on_errors=on_errors,
            return_type=return_type)

    def _func(self, elems, nums):
        elems_ = self.elements.loc[elems, :]
        w_ = np.array(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['sum:' + s for s in self.elements]


class WeightedVarFeature(_CompositionalFeature):
    def __init__(self, elements=None, *, n_jobs=-1, include=None,
                 exclude=None, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
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

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude, on_errors=on_errors,
            return_type=return_type)

    def _func(self, elems, nums):
        elems_ = self.elements.loc[elems, :]
        w_ = nums / np.sum(nums)
        mean_ = w_.dot(elems_)
        var_ = elems_ - mean_
        return w_.dot(var_ ** 2)

    @property
    def feature_labels(self):
        return ['var:' + s for s in self.elements]


class MaxFeature(_CompositionalFeature):
    def __init__(self, elements=None, *, n_jobs=-1, include=None,
                 exclude=None, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
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

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude, on_errors=on_errors,
            return_type=return_type)

    def _func(self, elems, _):
        elems_ = self.elements.loc[elems, :]
        return elems_.max().values

    @property
    def feature_labels(self):
        return ['max:' + s for s in self.elements]


class MinFeature(_CompositionalFeature):
    def __init__(self, elements=None, *, n_jobs=-1, include=None,
                 exclude=None, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
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

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude, on_errors=on_errors,
            return_type=return_type)

    def _func(self, elems, _):
        elems_ = self.elements.loc[elems, :]
        return elems_.min().values

    @property
    def feature_labels(self):
        return ['min:' + s for s in self.elements]


class Compositions(BaseDescriptor):
    """
    Calculate elemental descriptors from compound's composition.
    """

    def __init__(self, elements=None, *, n_jobs=-1, featurizers='all', on_errors='nan'):
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
        if elements is None:
            elements = preset.elements_completed

        self.composition = WeightedAvgFeature(elements, n_jobs=n_jobs, on_errors=on_errors)
        self.composition = WeightedSumFeature(elements, n_jobs=n_jobs, on_errors=on_errors)
        self.composition = WeightedVarFeature(elements, n_jobs=n_jobs, on_errors=on_errors)
        self.composition = MaxFeature(elements, n_jobs=n_jobs, on_errors=on_errors)
        self.composition = MinFeature(elements, n_jobs=n_jobs, on_errors=on_errors)
