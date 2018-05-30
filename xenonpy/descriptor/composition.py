# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
from pymatgen import Composition

from .base import BaseFeaturizer, BaseDescriptor
from ..datatools.dataset import Loader


class WeightedAvgFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, elements=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(n_jobs=n_jobs)
        self.__authors__ = ['TsumiNa']

        if elements is not None:
            self.elemental_info = elements
        else:
            self.elemental_info = Loader().elements_completed

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, Composition):
            comp = comp.as_dict()
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        elems_ = self.elemental_info.loc[elems_, :]
        w_nums = nums_ / np.sum(nums_)
        return w_nums.dot(elems_)

    @property
    def feature_labels(self):
        return ['ave:' + s for s in self.elemental_info]


class WeightedSumFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, elements=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(n_jobs=n_jobs)
        self.__authors__ = ['TsumiNa']

        if elements is not None:
            self.elemental_info = elements
        else:
            self.elemental_info = Loader().elements_completed

    def featurize(self, comp):
        elements_, nums_ = [], []
        if isinstance(comp, Composition):
            comp = comp.as_dict()
        for e, n in comp.items():
            elements_.append(e)
            nums_.append(n)
        elements_ = self.elemental_info.loc[elements_, :]
        nums_ = np.array(nums_)
        return nums_.dot(elements_)

    @property
    def feature_labels(self):
        return ['sum:' + s for s in self.elemental_info]


class WeightedVarFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, elements=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(n_jobs=n_jobs)
        self.__authors__ = ['TsumiNa']

        if elements is not None:
            self.elemental_info = elements
        else:
            self.elemental_info = Loader().elements_completed

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, Composition):
            comp = comp.as_dict()
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        elems_ = self.elemental_info.loc[elems_, :]
        w_nums = nums_ / np.sum(nums_)
        e_mean_ = w_nums.dot(elems_)
        cen_elems = elems_ - e_mean_
        return w_nums.dot(cen_elems ** 2)

    @property
    def feature_labels(self):
        return ['var:' + s for s in self.elemental_info]


class MaxFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, elements=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(n_jobs=n_jobs)
        self.__authors__ = ['TsumiNa']

        if elements is not None:
            self.elemental_info = elements
        else:
            self.elemental_info = Loader().elements_completed

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, Composition):
            comp = comp.as_dict()
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        elems_ = self.elemental_info.loc[elems_, :]
        return elems_.max().values

    @property
    def feature_labels(self):
        return ['max:' + s for s in self.elemental_info]


class MinFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, elements=None):
        """

        Parameters
        ----------
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        """

        super().__init__(n_jobs=n_jobs)
        self.__authors__ = ['TsumiNa']

        if elements is not None:
            self.elemental_info = elements
        else:
            self.elemental_info = Loader().elements_completed

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, Composition):
            comp = comp.as_dict()
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        elems_ = self.elemental_info.loc[elems_, :]
        return elems_.min().values

    @property
    def feature_labels(self):
        return ['min:' + s for s in self.elemental_info]


class CompositionDescriptor(BaseDescriptor):
    """
    Calculate elemental descriptors from compound's composition.
    """

    def __init__(self,
                 n_jobs=-1,
                 elemental_info=None
                 ):
        """

        Parameters
        ----------
        methods: str
            Calculation method(s) which to be used must in the :attr:`methods` list.
        elemental_info: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        comps_col: str
            Column's name which have compound's ``composition``.
            ``composition`` described in dict object.
        """

        if elemental_info is not None:
            self._elemental_info = elemental_info
        else:
            self._elemental_info = Loader().elements_completed

        self.composition = WeightedAvgFeature(n_jobs, self._elemental_info)
        self.composition = WeightedSumFeature(n_jobs, self._elemental_info)
        self.composition = WeightedVarFeature(n_jobs, self._elemental_info)
        self.composition = MaxFeature(n_jobs, self._elemental_info)
        self.composition = MinFeature(n_jobs, self._elemental_info)
