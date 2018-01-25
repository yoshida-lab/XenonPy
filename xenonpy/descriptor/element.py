# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pymatgen import Composition
from sklearn.base import TransformerMixin, BaseEstimator

from ..utils.datatools import Loader


# from ..pipeline import combinator


class ElementDesc(BaseEstimator, TransformerMixin):
    """
    Calculate elemental descriptors from compound's composition. i.e:

        >>> from xenonpy.descriptor import ElementDesc
        >>> Desc = ElementDesc()
        >>> desc = Desc.fit_transform(compounds)
    """

    #: methods can be used to calculate descriptors.
    methods = {'weighted_average', 'weighted_sum', 'weighted_variance', 'max', 'min'}

    def __init__(self,
                 *methods: str,
                 elements=None,
                 comps_col: str = 'composition',
                 verbose: bool = True
                 ):
        """

        Parameters
        ----------
        methods: str
            Calculation method(s) which to be used must in the :attr:`methods` list.
        elements: panda.DataFrame
            Elements information in `pandas.DataFrame` object. indexed by element symbol.
        comps_col: str
            Column's name which have compound's ``composition``.
            ``composition`` described in dict object.
        verbose: bool
            If verbose output.
        """

        if methods:
            if not set(methods).issubset(self.methods):
                raise ValueError(
                    'method(s) must be as "{:s}", left it None to use all'.
                        format(', '.join(self.methods)))
            self.methods = methods

        if elements:
            self.elems = elements
        else:
            # csv_path = str(Path(__file__).parent / 'completed.csv')
            self.elems = Loader().elements_completed

        self.verbose = verbose
        self.comps_col = comps_col
        self.labs = []
        self.funcs = []

    def fit(self, _):
        raw_labs = self.elems.columns.tolist()

        def ave(nums, elems):
            w_nums = nums / np.sum(nums)
            return w_nums.dot(elems)

        def sum_(nums, elems):
            return nums.dot(elems)

        def var(nums, elems):
            w_nums = nums / np.sum(nums)
            e_mean = w_nums.dot(elems)
            cen_elems = elems - e_mean
            return nums.dot(cen_elems ** 2)

        def max_(_, elems):
            return elems.max()

        def min_(_, elems):
            return elems.min()

        if 'weighted_average' in self.methods:
            self.labs += ['ave:' + s for s in raw_labs]
            self.funcs.append(ave)
        if 'weighted_sum' in self.methods:
            self.labs += ['sum:' + s for s in raw_labs]
            self.funcs.append(sum_)
        if 'weighted_variance' in self.methods:
            self.labs += ['var:' + s for s in raw_labs]
            self.funcs.append(var)
        if 'max' in self.methods:
            self.labs += ['max:' + s for s in raw_labs]
            self.funcs.append(max_)
        if 'min' in self.methods:
            self.labs += ['min:' + s for s in raw_labs]
            self.funcs.append(min_)

        return self

    def transform(self, cpds):
        if self.comps_col not in cpds:
            raise ValueError('need composition to calculate descriptor')

        def mk_desc(comp):
            # comp = comps[0]
            elems, nums = [], []
            if isinstance(comp, Composition):
                comp = comp.as_dict()
            for e, n in comp.items():
                elems.append(e)
                nums.append(n)
            nums = np.array(nums)
            elems = self.elems.loc[elems, :]
            ret = np.array([])
            for func in self.funcs:
                ret = np.concatenate((ret, func(nums, elems)))
            return ret

        if self.verbose:
            print('converting {:d} compounds'.format(cpds.shape[0]))
        ret = cpds[self.comps_col].apply(mk_desc).tolist()
        return pd.DataFrame(np.array(ret), index=cpds.index, columns=self.labs)
