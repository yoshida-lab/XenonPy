#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pytest
from pandas import DataFrame, Series

from xenonpy.datatools.preprocess import Splitter


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
    list_ = list(range(10))
    df = DataFrame(matrix)
    se = Series(array)
    flag = list('abcba') * 2

    yield array, matrix, list_, df, se, flag

    print('test over')


def test_data_splitter_2():
    ds = Splitter(10)
    assert ds.size == 10
    train, test = ds.split()
    assert train.size == 8
    assert test.size == 2


def test_roll_1():
    ds = Splitter(10, test_size=0.3)
    train, test = ds.split()
    assert train.size == 7
    assert test.size == 3
    ds.roll(test_size=0.4)
    train, test = ds.split()
    assert train.size == 6
    assert test.size == 4


def test_roll_2():
    ds = Splitter(10, test_size=0.3)
    train, test = ds.split()
    assert train.size == 7
    assert test.size == 3
    ds.roll(test_size=0.3)
    train_, test_ = ds.split()
    assert not np.array_equal(train, train_)
    assert not np.array_equal(test, test_)


def test_split_1(data):
    ds = Splitter(10, test_size=0.3)
    train, test = ds.split(data[0])
    for d in train:
        assert d in data[0]
    for d in test:
        assert d in data[0]


def test_split_2(data):
    ds = Splitter(10, test_size=0.1)
    try:
        ds.split(data[1][1:])
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'
    _, test = ds.split(data[1])
    assert test[0] in data[1]


def test_split_3(data):
    ds = Splitter(10)
    try:
        ds.split(data[0])
        ds.split(data[1])
        ds.split(data[2])
        ds.split(data[3])
        ds.split(data[4])
    except TypeError:
        assert False, 'should not got TypeError'
    else:
        assert True

    try:
        ds.split([1])
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'

    try:
        ds.split('ss')
    except TypeError:
        assert True
    else:
        assert False, 'should got TypeError'


def test_split_4(data):
    ds = Splitter(10)
    x, x_, y, y_ = ds.split(data[3], data[4])
    assert isinstance(x, np.ndarray)


def test_cv_1(data):
    ds = Splitter(10, test_size=0, cv=5)
    cv = ds.cv()
    tmp = []
    for x, x_ in cv:
        assert isinstance(x, np.ndarray)
        assert isinstance(x_, np.ndarray)
        assert x.size + x_.size == 10
        tmp.append(x_)
    tmp = np.concatenate(tmp)
    tmp = np.sort(tmp)
    assert np.array_equal(tmp, data[0])


def test_cv_2(data):
    ds = Splitter(10, test_size=0.2, cv=4)
    cv = ds.cv()
    tmp = []
    tmp_x_ = []
    for x, x_, _x_ in cv:
        assert isinstance(x, np.ndarray)
        assert isinstance(x_, np.ndarray)
        assert isinstance(_x_, np.ndarray)
        assert x.size + x_.size == 8
        assert _x_.size == 2
        tmp_x_.append(_x_)
        tmp.append(x_)
    assert np.array_equal(tmp_x_[1], tmp_x_[3])
    tmp = np.concatenate(tmp)
    assert tmp.size == 8
    tmp = np.concatenate([tmp, _x_])
    tmp = np.sort(tmp)
    assert np.array_equal(tmp, data[0])


def test_cv_3(data):
    ds = Splitter(10, test_size=0, cv=data[5])
    cv = ds.cv()
    tmp = []
    for x, x_ in cv:
        assert isinstance(x, np.ndarray)
        assert isinstance(x_, np.ndarray)
        assert x.size + x_.size == 10
        tmp.append(x_)
    sizes = np.sort([x.size for x in tmp])
    assert np.array_equal(sizes, [2, 4, 4])
    tmp = np.concatenate(tmp)
    tmp = np.sort(tmp)
    assert np.array_equal(tmp, data[0])


if __name__ == "__main__":
    pytest.main()
