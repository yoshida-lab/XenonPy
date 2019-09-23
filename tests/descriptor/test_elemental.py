#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import pytest

from xenonpy.descriptor import Compositions, Counting
from xenonpy.descriptor.base import BaseCompositionFeaturizer


def test_compositional_feature_1():
    class FakeFeaturizer(BaseCompositionFeaturizer):
        @property
        def feature_labels(self):
            return ['min:' + s for s in self._elements]

        def mix_function(self, elems, nums):
            elems_ = self._elements.loc[elems, :]
            w_ = nums / np.sum(nums)
            return w_.dot(elems_)

    desc = FakeFeaturizer(n_jobs=1)
    tmp = desc.fit_transform([{'H': 2}])

    assert isinstance(tmp, list)

    with pytest.raises(KeyError):
        desc.fit_transform([{'Bl': 2}])

    desc = FakeFeaturizer(n_jobs=1, on_errors='nan')
    tmp = desc.fit_transform([{'Bl': 2}])
    assert np.all(np.isnan(tmp[0]))


def test_counting_feature_1():
    comps = [{'H': 2, 'He': 1}, {'Li': 1}]
    counting = Counting(return_type='array')
    tmp = counting.transform(comps)
    assert tmp.shape == (2, 94)

    assert np.all(tmp[0, :2] == np.array([2, 1]))
    assert np.all(tmp[0, 2:] == np.zeros(92))
    assert np.all(tmp[1, :3] == np.array([0, 0, 1]))

    ohv = Counting(return_type='array', one_hot_vec=True)
    tmp = ohv.transform(comps)
    assert tmp.shape == (2, 94)

    assert np.all(tmp[0, :2] == np.array([1, 1]))
    assert np.all(tmp[0, 2:] == np.zeros(92))
    assert np.all(tmp[1, :3] == np.array([0, 0, 1]))


def test_comp_descriptor_1():
    desc = Compositions(n_jobs=1)

    desc.fit_transform(pd.Series([{'H': 2}], name='composition'))
    desc.fit_transform(pd.Series([{'H': 2}], name='other'))

    tmp1 = desc.fit_transform(pd.Series([{'H': 2}], name='other'), composition='other')
    tmp2 = desc.fit_transform([{'H': 2}])

    assert tmp1.shape == (1, 290)
    assert isinstance(tmp1, pd.DataFrame)
    assert isinstance(tmp2, pd.DataFrame)

    assert np.all(tmp1.values == tmp2.values)

    tmp = desc.transform([{'H': 2}], featurizers=['WeightedAverage'])
    assert tmp.shape == (1, 58)

    tmp = desc.transform([{'H': 2}], featurizers='all')
    assert tmp.shape == (1, 500)

    tmp = desc.transform([{'H': 2}], featurizers=Compositions.classic)
    assert tmp.shape == (1, 290)


if __name__ == "__main__":
    pytest.main()
