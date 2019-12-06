#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from torch import nn

from xenonpy.descriptor import FrozenFeaturizer
from xenonpy.model import SequentialLinear


@pytest.fixture(scope='module')
def data():
    # prepare path
    model1 = nn.Sequential(
        nn.Sequential(OrderedDict(layer=nn.Linear(10, 7), act_func=nn.ReLU())),
        nn.Sequential(OrderedDict(layer=nn.Linear(7, 4), act_func=nn.ReLU())),
        nn.Sequential(OrderedDict(layer=nn.Linear(4, 1))),
    )
    model2 = SequentialLinear(10, 1, h_neurons=(7, 4))
    descriptor1 = np.random.rand(10, 10)
    descriptor2 = np.random.rand(20, 10)
    index = ['id_' + str(i) for i in range(10)]
    df = pd.DataFrame(descriptor1, index=index)

    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield model1, descriptor1, df, index, descriptor2, model2

    print('test over')


def test_frozen_featurizer_1(data):
    ff = FrozenFeaturizer(data[0])
    ff_features = ff.fit_transform(data[1])
    assert ff_features.shape == (10, 11)
    ff_features = ff.fit_transform(data[1], depth=1)
    assert ff_features.shape == (10, 4)
    ff_features = ff.fit_transform(data[1])
    assert ff_features.shape == (10, 11)

    ff = FrozenFeaturizer(data[5])
    ff_features = ff.fit_transform(data[1])
    assert ff_features.shape == (10, 11)
    ff_features = ff.fit_transform(data[1], depth=1)
    assert ff_features.shape == (10, 4)
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

    ff = FrozenFeaturizer(data[5])
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
    with pytest.raises(ValueError):
        ff.feature_labels
    ff.fit_transform(data[2])
    ff.feature_labels

    ff = FrozenFeaturizer(data[5])
    with pytest.raises(ValueError):
        ff.feature_labels
    ff.fit_transform(data[2])
    ff.feature_labels


def test_frozen_featurizer_4(data):
    ff = FrozenFeaturizer(data[0])
    ff_features = ff.fit_transform(data[1])
    assert ff_features.shape == (10, 11)
    ff_features = ff.fit_transform(data[4])
    assert ff_features.shape == (20, 11)

    ff = FrozenFeaturizer(data[5])
    ff_features = ff.fit_transform(data[1])
    assert ff_features.shape == (10, 11)
    ff_features = ff.fit_transform(data[4])
    assert ff_features.shape == (20, 11)


def test_frozen_featurizer_5(data):
    ff = FrozenFeaturizer(data[0])

    with pytest.warns(UserWarning, match='is over the max depth of hidden layers starting at the given'):
        ff_features = ff.fit_transform(data[1], depth=2, n_layer=3)
        assert ff_features.shape == (10, 11)
    with pytest.warns(UserWarning, match='is greater than the max depth of hidden layers'):
        ff_features = ff.fit_transform(data[4], depth=3, n_layer=2)
        assert ff_features.shape == (20, 11)

    ff = FrozenFeaturizer(data[5])
    ff_features = ff.transform(data[1], depth=2, return_type='df')

    desc_ori = []
    for i in [1, 2]:
        desc_ori.append(ff_features[ff_features.columns[[f'L(-{i})' in s for s in ff_features.columns.values]]])

    desc_new = []
    for i in [1, 2]:
        for j in [1, 2]:
            desc_new.append(ff.transform(data[1], depth=j, n_layer=i, return_type='df'))

    for i in range(2):
        tmp = desc_new[i] == desc_ori[i]
        assert all(tmp.all())


if __name__ == "__main__":
    pytest.main()
