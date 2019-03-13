#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import pytest

from xenonpy.descriptor.base import BaseDescriptor, BaseFeaturizer


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    class _FakeFeaturier(BaseFeaturizer):

        def __init__(self, n_jobs=1):
            super().__init__(n_jobs=n_jobs)

        def featurize(self, *x):
            return x[0]

        @property
        def feature_labels(self):
            return ['labels']

    class _FakeDescriptor(BaseDescriptor):

        def __init__(self):
            super().__init__()
            self.g1 = _FakeFeaturier()
            self.g1 = _FakeFeaturier()
            self.g2 = _FakeFeaturier()

    # prepare test data
    yield dict(featurizer=_FakeFeaturier, descriptor=_FakeDescriptor)

    print('test over')


def test_base_feature_props(data):
    bf = BaseFeaturizer()

    # test n_jobs
    assert bf.n_jobs == cpu_count()
    bf.n_jobs = 1
    assert bf.n_jobs == 1
    bf.n_jobs = 100
    assert bf.n_jobs == cpu_count()

    # test citation, author
    assert bf.citations == 'No citations'
    assert bf.authors == 'anonymous'

    # test labels, featurize
    with pytest.raises(NotImplementedError):
        bf.featurize(1)

    with pytest.raises(NotImplementedError):
        bf.feature_labels


def test_base_feature_1(data):
    featurizer = data['featurizer']()
    assert isinstance(featurizer, BaseFeaturizer)
    assert featurizer.n_jobs == 1
    assert featurizer.featurize(10) == 10
    assert featurizer.feature_labels == ['labels']
    with pytest.raises(TypeError):
        featurizer.fit_transform(56)


def test_base_feature_2(data):
    featurizer = data['featurizer']()
    ret = featurizer.fit_transform([1, 2, 3, 4])
    assert isinstance(ret, list)
    ret = featurizer.fit_transform([1, 2, 3, 4], return_type='array')
    assert isinstance(ret, np.ndarray)
    ret = featurizer.fit_transform([1, 2, 3, 4], return_type='df')
    assert isinstance(ret, pd.DataFrame)
    ret = featurizer.fit_transform(np.array([1, 2, 3, 4]))
    assert isinstance(ret, np.ndarray)
    assert (ret == np.array([1, 2, 3, 4])).all()
    ret = featurizer.fit_transform(pd.Series([1, 2, 3, 4]))
    assert isinstance(ret, pd.DataFrame)
    assert (ret.values.ravel() == np.array([1, 2, 3, 4])).all()


def test_base_feature_3(data):
    class _ErrorFeaturier(BaseFeaturizer):

        def __init__(self, n_jobs=1, on_errors='raise'):
            super().__init__(n_jobs=n_jobs, on_errors=on_errors)

        def featurize(self, *x):
            raise ValueError()

        @property
        def feature_labels(self):
            return ['labels']

    featurizer = _ErrorFeaturier()
    assert isinstance(featurizer, BaseFeaturizer)
    with pytest.raises(ValueError):
        featurizer.fit_transform([1, 2, 3, 4])

    featurizer = _ErrorFeaturier(on_errors='keep')
    tmp = featurizer.fit_transform([1, 2, 3, 4])
    assert np.alltrue([isinstance(e[0], ValueError) for e in tmp])

    featurizer = _ErrorFeaturier(on_errors='nan')
    tmp = featurizer.fit_transform([1, 2, 3, 4])
    assert np.alltrue([np.isnan(e[0]) for e in tmp])


def test_base_descriptor_1(data):
    bd = BaseDescriptor()

    # test n_jobs
    assert bd.elapsed == 0
    assert bd.n_jobs == 1
    bd.n_jobs = 100
    assert bd.n_jobs == cpu_count()

    # test featurizers list
    assert hasattr(bd, '__featurizers__')
    assert not bd.__featurizers__


def test_base_descriptor_2(data):
    bd = data['descriptor']()
    assert len(bd.__featurizers__) == 2
    assert 'g1' in bd.__featurizers__
    assert 'g2' in bd.__featurizers__


def test_base_descriptor_3(data):
    bd = data['descriptor']()
    with pytest.raises(TypeError):
        bd.fit([1, 2, 3, 4]),


def test_base_descriptor_4(data):
    feature = data['featurizer']

    class _FakeDescriptor(BaseDescriptor):

        def __init__(self):
            super().__init__()
            self.g1 = feature()
            self.g1 = feature()

    bd = _FakeDescriptor()
    bd.fit([1, 2, 3, 4])


def test_base_descriptor_5(data):
    bd = data['descriptor']()
    x = pd.DataFrame({'g1': [1, 2, 3, 4], 'g3': [1, 2, 3, 4]})
    tmp = bd.fit_transform(x)
    assert isinstance(tmp, pd.DataFrame)
    assert np.all(tmp.values == np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))

    x = pd.Series([1, 2, 3, 4], name='g1')
    tmp = bd.transform(x)
    assert isinstance(tmp, pd.DataFrame)
    assert np.all(tmp.values == np.array([[1, 1], [2, 2], [3, 3], [4, 4]]))


def test_base_descriptor_6(data):
    bd = data['descriptor']()
    x = pd.DataFrame({'g3': [1, 2, 3, 4], 'g4': [1, 2, 3, 4]})
    with pytest.raises(KeyError):
        bd.fit_transform(x)

    x = pd.Series([1, 2, 3, 4], name='g3')
    with pytest.raises(KeyError):
        bd.fit_transform(x)


if __name__ == "__main__":
    pytest.main()
