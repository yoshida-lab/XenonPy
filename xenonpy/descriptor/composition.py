# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
from pymatgen.core.composition import Composition as PMGComp

from .base import BaseFeaturizer, BaseDescriptor
from ..datatools.dataset import preset


class _CompositionalFeature(BaseFeaturizer):
    def __init__(self, n_jobs=-1, *, elements=None, include=None,
                 exclude=None):
        """
        Base class for composition feature.
        """

        if include and exclude:
            raise ValueError(
                'Paratemer "include" and "exclude" are mutually exclusive.')
        super().__init__(n_jobs=n_jobs)

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


class WeightedAvgFeature(_CompositionalFeature):
    def __init__(self, n_jobs=-1, *, elements=None, include=None,
                 exclude=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude)

    def _func(self, elems, nums):
        elems_ = self.elements.loc[elems, :]
        w_ = nums / np.sum(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['ave:' + s for s in self.elements]


class WeightedSumFeature(_CompositionalFeature):
    def __init__(self, n_jobs=-1, *, elements=None, include=None,
                 exclude=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude)

    def _func(self, elems, nums):
        elems_ = self.elements.loc[elems, :]
        w_ = np.array(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['sum:' + s for s in self.elements]


class WeightedVarFeature(_CompositionalFeature):
    def __init__(self, n_jobs=-1, *, elements=None, include=None,
                 exclude=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude)

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
    def __init__(self, n_jobs=-1, *, elements=None, include=None,
                 exclude=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude)

    def _func(self, elems, _):
        elems_ = self.elements.loc[elems, :]
        return elems_.max().values

    @property
    def feature_labels(self):
        return ['max:' + s for s in self.elements]


class MinFeature(_CompositionalFeature):
    def __init__(self, n_jobs=-1, *, elements=None, include=None,
                 exclude=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(
            n_jobs=n_jobs, elements=elements, include=include, exclude=exclude)

    def _func(self, elems, _):
        elems_ = self.elements.loc[elems, :]
        return elems_.min().values

    @property
    def feature_labels(self):
        return ['min:' + s for s in self.elements]


class Composition(BaseDescriptor):
    """
    Calculate elemental descriptors from compound's composition.
    """

    def __init__(self, n_jobs=-1, *, elements=None, include=None,
                 exclude=None):
        """

        Parameters
        ----------
        methods: str
            Calculation method(s) which to be used must in the :attr:`methods` list.
        elemental: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        include: list
            Column's names of elemental info that should be used in descriptor calculation.
        exclude: list
            Column's names of elemental info that should not be used in descriptor calculation.
        """

        super().__init__()
        self.n_jobs = n_jobs
        if elements is None:
            elements = preset.elements_completed
        if include is not None:
            elements = elements[include]
        if exclude is not None:
            elements = elements.drop(exclude, axis=1)
        self.elements = elements

        self.composition = WeightedAvgFeature(n_jobs, elements=elements)
        self.composition = WeightedSumFeature(n_jobs, elements=elements)
        self.composition = WeightedVarFeature(n_jobs, elements=elements)
        self.composition = MaxFeature(n_jobs, elements=elements)
        self.composition = MinFeature(n_jobs, elements=elements)
