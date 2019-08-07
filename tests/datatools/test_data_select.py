#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame, Series

from xenonpy.datatools.splitter import Splitter


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    # prepare test data
    array = np.arange(10)
    matrix = np.arange(100).reshape(10, 10)
    df = DataFrame(matrix)
    se = Series(array)
    flag = list('abcba') * 2

    yield array, matrix, df, se, flag

    print('test over')


def test_init_1():
    with pytest.raises(RuntimeError, match='<test_size> can be none only if <cv> is not none'):
        Splitter(10, test_size=0, k_fold=None)


def test_roll_1():
    sp = Splitter(10, test_size=0.3, random_state=123456)
    train, test = sp.split()
    assert train.size == 7
    assert test.size == 3
    train_, test_ = sp.split()
    assert train_.size == 7
    assert test_.size == 3
    assert np.array_equal(train_, train)
    assert np.array_equal(test_, test)

    sp.roll(random_state=123456)
    train_, test_ = sp.split()
    assert train_.size == 7
    assert test_.size == 3
    assert np.array_equal(train_, train)
    assert np.array_equal(test_, test)

    sp.roll()
    train_, test_ = sp.split()
    assert not np.array_equal(train_, train)
    assert not np.array_equal(test_, test)


def test_split_1(data):
    sp = Splitter(10)
    with pytest.raises(RuntimeError, match='parameter <cv> must be set'):
        for _ in sp.cv():
            pass

    assert sp.size == 10
    train, test = sp.split()
    assert train.size == 8
    assert test.size == 2

    train, test = sp.split(data[0])
    for d in train:
        assert d in data[0]
    for d in test:
        assert d in data[0]


def test_split_2(data):
    sp = Splitter(10, test_size=0.1)
    with pytest.raises(ValueError, match='parameters <arrays> must have size 10 for dim 0'):
        sp.split(data[1][1:])
    _, test = sp.split(data[1])
    assert test[0] in data[1]


def test_split_3(data):
    sp = Splitter(10)
    sp.split(data[0])
    sp.split(data[1])
    sp.split(data[2])
    sp.split(data[3])

    with pytest.raises(TypeError, match='<arrays> must be numpy.ndarray, pandas.DataFrame, or pandas.Series'):
        sp.split([1])


def test_split_4(data):
    sp = Splitter(10)
    x, x_, y, y_, z, z_ = sp.split(data[1], data[2], data[3])
    assert isinstance(x, np.ndarray)
    assert isinstance(x_, np.ndarray)
    assert isinstance(y, pd.DataFrame)
    assert isinstance(y_, pd.DataFrame)
    assert isinstance(z, pd.Series)
    assert isinstance(z_, pd.Series)


def test_cv_1(data):
    sp = Splitter(10, test_size=0, k_fold=5, random_state=123456)
    with pytest.raises(RuntimeError, match='split action is illegal because `test_size` is none'):
        sp.split()

    tmp = []
    for i, (x, x_) in enumerate(sp.cv()):
        assert x.size == 8
        assert x_.size == 2
        assert isinstance(x, np.ndarray)
        assert isinstance(x_, np.ndarray)
        tmp.append(x_)
    assert i == 4
    tmp = np.concatenate(tmp)
    assert not np.array_equal(tmp, data[0])
    tmp = np.sort(tmp)
    assert np.array_equal(tmp, data[0])

    tmp = []
    for x, x_ in sp.cv(less_for_train=True):
        assert x.size == 2
        assert x_.size == 8
        tmp.append(x)
    tmp = np.concatenate(tmp)
    tmp = np.sort(tmp)
    assert np.array_equal(tmp, data[0])


def test_cv_2(data):
    sp = Splitter(10, test_size=0.2, k_fold=4)
    tmp = []
    tmp_x_ = []
    for x, x_, _x_ in sp.cv():
        assert x.size == 6
        assert x_.size == 2
        assert _x_.size == 2
        assert isinstance(x, np.ndarray)
        assert isinstance(x_, np.ndarray)
        assert isinstance(_x_, np.ndarray)
        tmp_x_.append(_x_)
        tmp.append(x_)
    assert np.array_equal(tmp_x_[0], tmp_x_[1])
    assert np.array_equal(tmp_x_[0], tmp_x_[2])
    assert np.array_equal(tmp_x_[0], tmp_x_[3])
    tmp = np.concatenate(tmp)
    assert tmp.size == 8
    tmp = np.concatenate([tmp, tmp_x_[0]])
    tmp = np.sort(tmp)
    assert np.array_equal(tmp, data[0])


def test_cv_3(data):
    np.random.seed(123456)
    sp = Splitter(10, test_size=0, k_fold=data[4])
    tmp = []
    for x, x_ in sp.cv():
        assert isinstance(x, np.ndarray)
        assert isinstance(x_, np.ndarray)
        assert x.size + x_.size == 10
        tmp.append(x_)
    sizes = np.sort([x.size for x in tmp])
    assert np.array_equal(sizes, [2, 4, 4])
    tmp = np.concatenate(tmp)
    tmp = np.sort(tmp)
    assert np.array_equal(tmp, data[0])


def test_cv_4(data):
    sp = Splitter(10, test_size=0, k_fold=5, random_state=123456)
    tmp = []
    for _, x_ in sp.cv():
        tmp.append(x_)
    tmp = np.concatenate(tmp)

    tmp_ = []
    for _, x_ in sp.cv():
        tmp_.append(x_)
    tmp_ = np.concatenate(tmp_)
    assert np.array_equal(tmp, tmp_)

    tmp_ = []
    sp.roll()
    for _, x_ in sp.cv():
        tmp_.append(x_)
    tmp_ = np.concatenate(tmp_)
    assert not np.array_equal(tmp, tmp_)


def test_cv_5(data):
    sp = Splitter(10, test_size=0, k_fold=5)
    for _, x_, _, y_, _, z_ in sp.cv(data[1], data[2], data[3]):
        assert isinstance(x_, np.ndarray)
        assert isinstance(y_, pd.DataFrame)
        assert isinstance(z_, pd.Series)
        assert x_.size == 20
        assert y_.size == 20
        assert z_.size == 2


if __name__ == "__main__":
    pytest.main()
