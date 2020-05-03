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

    class _FakeFeaturier1(BaseFeaturizer):

        def __init__(self, n_jobs=1):
            super().__init__(n_jobs=n_jobs)

        def featurize(self, *x, **kwargs):
            return x[0]

        @property
        def feature_labels(self):
            return ['label1']

    class _FakeFeaturier2(BaseFeaturizer):

        def __init__(self, n_jobs=1):
            super().__init__(n_jobs=n_jobs)

        def featurize(self, *x, **kwargs):
            return x[0]

        @property
        def feature_labels(self):
            return ['label2']

    class _FakeFeaturier3(BaseFeaturizer):

        def __init__(self, n_jobs=1):
            super().__init__(n_jobs=n_jobs)

        def featurize(self, *x, **kwargs):
            return x[0]

        @property
        def feature_labels(self):
            return ['label3']

    class _FakeDescriptor(BaseDescriptor):

        def __init__(self, featurizers='all', on_errors='raise'):
            super().__init__(featurizers=featurizers, on_errors=on_errors)
            self.g1 = _FakeFeaturier1()
            self.g1 = _FakeFeaturier2()
            self.g2 = _FakeFeaturier3()

    # prepare test data
    yield dict(featurizer=_FakeFeaturier1, descriptor=_FakeDescriptor)

    print('test over')


def test_base_feature_props():

    class _FakeFeaturier(BaseFeaturizer):

        def __init__(self):
            super().__init__()

        def featurize(self, *x, **kwargs):
            return x[0]

        @property
        def feature_labels(self):
            return ['labels']

    bf = _FakeFeaturier()
    with pytest.raises(ValueError, match='`on_errors`'):
        bf.on_errors = 'illegal'

    with pytest.raises(ValueError, match='`return_type`'):
        bf.return_type = 'illegal'

    # test n_jobs
    assert bf.n_jobs == cpu_count()
    bf.n_jobs = 1
    assert bf.n_jobs == 1
    bf.n_jobs = 100
    assert bf.n_jobs == cpu_count()

    # test citation, author
    assert bf.citations == 'No citations'
    assert bf.authors == 'anonymous'


def test_base_feature_1(data):
    featurizer = data['featurizer']()
    assert isinstance(featurizer, BaseFeaturizer)
    assert featurizer.n_jobs == 1
    assert featurizer.featurize(10) == 10
    assert featurizer.feature_labels == ['label1']
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


def test_base_feature_para(data):
    featurizer = data['featurizer'](n_jobs=2)
    featurizer.parallel_verbose = 1
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

    # test featurizers list
    assert hasattr(bd, '__featurizers__')
    assert not bd.__featurizer_sets__

    with pytest.raises(ValueError, match='`on_errors`'):
        bd.on_errors = 'illegal'


def test_base_descriptor_2(data):
    bd = data['descriptor']()
    assert len(bd.all_featurizers) == 3
    assert 'g1' in bd.__featurizer_sets__
    assert 'g2' in bd.__featurizer_sets__

    assert bd.on_errors == 'raise'
    assert bd.__featurizer_sets__['g1'][0].on_errors == 'raise'
    bd.set_params(on_errors='nan')
    assert bd.on_errors == 'nan'
    assert bd.__featurizer_sets__['g1'][0].on_errors == 'nan'


def test_base_descriptor_3(data):
    bd = data['descriptor']()
    with pytest.raises(TypeError):
        bd.fit([1, 2, 3, 4]),


def test_base_descriptor_4(data):
    ff = data['featurizer']

    class FakeDescriptor(BaseDescriptor):

        def __init__(self):
            super().__init__()
            self.g1 = ff()
            self.g1 = ff()

    with pytest.raises(RuntimeError):
        FakeDescriptor()


def test_base_descriptor_5(data):
    bd = data['descriptor']()
    x = pd.DataFrame({'g1': [1, 2, 3, 4], 'g3': [10, 20, 30, 40]})
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
        bd.fit(x)
    with pytest.raises(KeyError):
        bd.transform(x)
    bd.fit(x, g1='g3', g2='g4')
    bd.transform(x)

    x = pd.DataFrame({'g1': [1, 2, 3, 4], 'g2': [1, 2, 3, 4]})
    with pytest.raises(KeyError):
        bd.transform(x)
    bd.transform(x, g3='g1', g4='g2')
    with pytest.raises(KeyError):
        bd.transform(x)
    bd.fit_transform(x, g3='g1', g4='g2')


def test_base_descriptor_7(data):
    bd = data['descriptor']()
    x = pd.Series([1, 2, 3, 4], name='g3')
    with pytest.raises(KeyError):
        bd.fit(x)
    with pytest.raises(KeyError):
        bd.transform(x)
    bd.fit(x, g1='g3')
    bd.transform(x)

    x = pd.Series([1, 2, 3, 4], name='g1')
    with pytest.raises(KeyError):
        bd.transform(x)
    bd.transform(x, g3='g1')
    with pytest.raises(KeyError):
        bd.transform(x)
    bd.fit_transform(x, g3='g1')


def test_base_descriptor_8(data):
    x = pd.DataFrame({'g1': [1, 2, 3, 4], 'g2': [10, 20, 30, 40]})

    bd = data['descriptor'](featurizers='_FakeFeaturier1')
    tmp = bd.transform(x)
    assert bd.featurizers == ('_FakeFeaturier1',)
    assert tmp.shape == (4, 1)
    assert tmp.columns == ['label1']
    assert np.all(tmp.values == np.array([[1], [2], [3], [4]]))

    bd = data['descriptor'](featurizers=['_FakeFeaturier3'])
    tmp = bd.transform(x)
    assert bd.featurizers == ('_FakeFeaturier3',)
    assert tmp.shape == (4, 1)
    assert tmp.columns == ['label3']
    assert np.all(tmp.values == np.array([[10], [20], [30], [40]]))

    bd = data['descriptor'](featurizers=['_FakeFeaturier1', '_FakeFeaturier3'])
    tmp = bd.transform(x)
    assert tmp.shape == (4, 2)
    assert tmp.columns.tolist() == ['label1', 'label3']
    assert bd.featurizers == ('_FakeFeaturier1', '_FakeFeaturier3')
    assert np.all(tmp.values == np.array([[1, 10], [2, 20], [3, 30], [4, 40]]))

    bd = data['descriptor']()
    # use all featurizers by default
    tmp = bd.fit_transform(x)
    assert tmp.shape == (4, 3)
    assert tmp.columns.tolist() == ['label1', 'label2', 'label3']
    assert np.all(tmp.values == np.array([[1, 1, 10], [2, 2, 20], [3, 3, 30], [4, 4, 40]]))


if __name__ == "__main__":
    pytest.main()
