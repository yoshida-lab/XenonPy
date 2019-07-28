#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import os
from pathlib import Path
from shutil import rmtree

import pytest
from sklearn.externals import joblib as jl

from xenonpy.datatools import Storage


@pytest.fixture(scope='module')
def data():
    # prepare path
    dir_ = os.path.dirname(os.path.abspath(__file__))
    name = 'test'
    storage = Storage(f'{dir_}/{name}')

    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    storage(list('abcd'), list('efgh'))
    storage(key1=list('asdf'), key2=list('qwer'))
    storage(list('asdf'), key1=list('qwer'))

    yield storage, dir_, name

    rmtree(f'{dir_}/{name}', ignore_errors=True)
    rmtree(f'{dir_}/dump', ignore_errors=True)
    print('test over')


def test_storage_1(data):
    storage, dir_, name = data[0], data[1], data[2]
    assert str(storage) == '<{}> includes:\n"unnamed": 3\n"key1": 2\n"key2": 1'.format(name)
    assert len(storage._files['unnamed']) == 3, 'should got 3 files'
    assert len(storage._files['key1']) == 2, 'should got 1 files'
    assert len(storage._files['key2']) == 1, 'should got 1 files'


def test_storage_last_1(data):
    storage, dir_, name = data[0], data[1], data[2]

    last = storage.last()
    assert last == list('asdf'), 'retriever same data'

    last = storage.last('key1')
    assert last == list('qwer'), 'retriever same data'

    item = storage[:]
    assert item[1] == list('efgh'), 'retriever same data'
    item = storage[1]
    assert item == list('efgh'), 'retriever same data'

    item = storage['key2', :]
    assert item[0] == list('qwer'), 'retriever same data'
    item = storage['key1', 1]
    assert item == list('qwer'), 'retriever same data'


def test_dump_1(data):
    storage, dir_, name = data[0], data[1], data[2]

    path = f'{dir_}/dump'
    path_ = storage.dump(path, with_datetime=False)
    assert path_ == f'{path}/{name}.pkl.z'
    assert Path(path_).exists()

    dumped = jl.load(path_)
    assert dumped['key1'] == list('qwer')
    assert dumped['key2'] == list('qwer')
    assert dumped['unnamed'] == list('asdf')


def test_storage_chained(data):
    storage, dir_, name = data[0], data[1], data[2]
    with pytest.raises(AttributeError):
        storage.chain

    sub = storage.chain_
    assert sub.path == f'{storage.path}/chain'
    assert Path(sub.path).exists()
    assert hasattr(storage, 'chain_')


def test_storage_delete_1(data):
    storage, dir_, name = data[0], data[1], data[2]

    storage.rm(0)
    assert len(storage._files['unnamed']) == 2

    storage(key1=list('qwer'))
    assert len(storage._files['key1']) == 3

    storage.rm(slice(0, 2), 'key1')
    assert len(storage._files['key1']) == 1


def test_storage_clean_1(data):
    storage, dir_, name = data[0], data[1], data[2]

    storage.clean('key1')
    assert 'key1' not in storage._files, 'no storage dir'

    storage.clean()
    for _ in Path(storage.path).iterdir():
        assert False


if __name__ == '__main__':
    pytest.main()
