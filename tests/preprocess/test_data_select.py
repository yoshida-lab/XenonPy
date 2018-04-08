# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


import numpy as np
import pytest
from pandas import DataFrame, Series

from xenonpy.preprocess.data_select import DataSplitter


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    # prepare test data
    array = np.arange(10)
    matrix = np.matrix(np.arange(100).reshape(10, 10))
    list_ = list(range(10))
    df = DataFrame(matrix)
    se = Series(array)

    yield array, matrix, list_, df, se

    print('test over')


def test_data_splitter1(data):
    try:
        DataSplitter(data[0])
        DataSplitter(data[1])
        DataSplitter(data[2])
        DataSplitter(data[3])
        DataSplitter(data[4])
    except TypeError:
        assert False, 'should not got TypeError'
    else:
        assert True

    try:
        DataSplitter([1])
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'

    try:
        DataSplitter('ss')
    except TypeError:
        assert True
    else:
        assert False, 'should got TypeError'


def test_data_splitter2(data):
    ds = DataSplitter(data[0])
    assert ds.size == 10
    assert ds.shape == (10,)
    train, test = ds.index
    assert train.size == 8
    assert test.size == 2


def test_re_sample1(data):
    ds = DataSplitter(data[0], test_size=0.3)
    train, test = ds.index
    assert train.size == 7
    assert test.size == 3
    ds.re_sample(test_size=0.4)
    train, test = ds.index
    assert train.size == 6
    assert test.size == 4


def test_re_sample2(data):
    ds = DataSplitter(data[0], test_size=0.3)
    train, test = ds.index
    assert train.size == 7
    assert test.size == 3
    ds.re_sample(test_size=0.3)
    train_, test_ = ds.index
    assert not np.isin(train, train_).all()
    assert not np.isin(test, test_).all()


def test_split_data1(data):
    ds = DataSplitter(data[0], test_size=0.3)
    train, test = ds.split_data(data[0])
    for d in train:
        assert d in data[0]
    for d in test:
        assert d in data[0]


def test_split_data2(data):
    ds = DataSplitter(data[0], test_size=0.1)
    try:
        ds.split_data(data[1][1:])
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'
    _, test = ds.split_data(data[1])
    assert test[0] in data[1]


def test_split_data3(data):
    ds = DataSplitter(data[0])
    try:
        ds.split_data(data[0])
        ds.split_data(data[1])
        ds.split_data(data[2])
        ds.split_data(data[3])
        ds.split_data(data[4])
    except TypeError:
        assert False, 'should not got TypeError'
    else:
        assert True

    try:
        ds.split_data([1])
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'

    try:
        ds.split_data('ss')
    except TypeError:
        assert True
    else:
        assert False, 'should got TypeError'


def test_split_data4(data):
    ds = DataSplitter(data[0])
    x, x_, y, y_ = ds.split_data(data[3], data[4])
    assert isinstance(x, DataFrame)
    assert isinstance(y, Series)
    _x, _y = ds.split_data(data[3], data[4], test=False)
    assert np.isin(x, _x).all()
    assert np.isin(y, _y).all()
    _x, _y = ds.split_data(data[3], data[4], train=False)
    assert np.isin(x_, _x).all()
    assert np.isin(y_, _y).all()


if __name__ == "__main__":
    pytest.main()
