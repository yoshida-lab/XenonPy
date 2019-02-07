#  Copyright 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import numpy as np
import pandas as pd
import pytest
from torch.nn import Sequential

from xenonpy.descriptor import FrozenFeaturizer
from xenonpy.model.nn import Layer1d


@pytest.fixture(scope='module')
def data():
    # prepare path
    model = Sequential(Layer1d(10, 7), Layer1d(7, 4), Layer1d(4, 1))
    descriptor1 = np.random.rand(10, 10)
    descriptor2 = np.random.rand(20, 10)
    index = ['id_' + str(i) for i in range(10)]
    df = pd.DataFrame(descriptor1, index=index)

    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield model, descriptor1, df, index, descriptor2

    print('test over')


def test_frozen_featurizer_1(data):
    ff = FrozenFeaturizer(data[0])
    ff_features = ff.fit_transform(data[1])
    assert ff_features.shape == (10, 11)


def test_frozen_featurizer_2(data):
    ff = FrozenFeaturizer(data[0])
    ff_features = ff.fit_transform(data[2])
    assert ff_features.shape == (10, 11)
    assert isinstance(ff_features, pd.DataFrame)
    assert ff_features.index.tolist() == list(data[3])

    _hlayers = [7, 4]
    labels = ['L(' + str(i - len(_hlayers)) + ')_' + str(j + 1)
              for i in range(2)
              for j in range(_hlayers[i])]
    assert ff_features.columns.tolist() == list(labels)


def test_frozen_featurizer_3(data):
    ff = FrozenFeaturizer(data[0])
    try:
        ff.feature_labels
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'
    ff.fit_transform(data[2])
    try:
        ff.feature_labels
    except ValueError:
        assert False


def test_frozen_featurizer_4(data):
    ff = FrozenFeaturizer(data[0])
    ff_features = ff.fit_transform(data[1])
    assert ff_features.shape == (10, 11)
    ff_features = ff.fit_transform(data[4])
    assert ff_features.shape == (20, 11)


if __name__ == "__main__":
    pytest.main()
