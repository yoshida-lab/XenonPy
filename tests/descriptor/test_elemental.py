#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import pytest

from xenonpy.descriptor import Compositions
from xenonpy.descriptor.composition import _CompositionalFeature


def test_compositional_feature_1():
    class FakeFeature(_CompositionalFeature):
        @property
        def feature_labels(self):
            return ['min:' + s for s in self.elements]

        def _func(self, elems, nums):
            elems_ = self.elements.loc[elems, :]
            w_ = nums / np.sum(nums)
            return w_.dot(elems_)

    desc = FakeFeature(n_jobs=1)
    tmp = desc.fit_transform([{'H': 2}])

    assert isinstance(tmp, list)

    with pytest.raises(KeyError):
        desc.fit_transform([{'Bl': 2}])

    desc = FakeFeature(n_jobs=1, on_errors='nan')
    tmp = desc.fit_transform([{'Bl': 2}])
    assert np.all(np.isnan(tmp[0]))


def test_comp_descriptor_1():
    desc = Compositions(n_jobs=1)
    tmp1 = desc.fit_transform(pd.Series([{'H': 2}], name='composition'))
    tmp2 = desc.fit_transform(pd.Series([{'H': 2}], name='other'), composition='other')
    tmp3 = desc.fit_transform([{'H': 2}])

    assert isinstance(tmp1, pd.DataFrame)
    assert isinstance(tmp2, pd.DataFrame)
    assert isinstance(tmp3, pd.DataFrame)

    assert np.all(tmp1.values == tmp2.values)
    assert np.all(tmp2.values == tmp3.values)


if __name__ == "__main__":
    pytest.main()
