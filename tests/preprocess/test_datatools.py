# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from os import remove
from pathlib import Path
from shutil import rmtree

import pytest
from sklearn.externals import joblib as jl

from xenonpy.preprocess.datatools import Loader, DataSet


@pytest.fixture(scope='module')
def setup():
    # prepare path
    test = dict(
        test_file=Path().cwd() / 'fetch_test.txt',
        fetch_file=
        'https://raw.githubusercontent.com/yoshida-lab/XenonPy/master/travis/fetch_test.txt',
        test_dir='test_dir',
        user_dataset='test_user_data')

    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield test

    if test['test_file'].exists():
        remove(str(test['test_file']))
    if (Path.home() / '.xenonpy/userdata' / test['test_dir']).exists():
        rmtree(str(Path.home() / '.xenonpy/userdata' / test['test_dir']))
    if (Path.home() / '.xenonpy/cached/travis').exists():
        rmtree(str(Path.home() / '.xenonpy/cached/travis'))
    if (Path().expanduser() / 'test_user_data.pkl.z').exists():
        remove(str(Path().expanduser() / 'test_user_data.pkl.z'))
    print('test over')


def test_loader():
    load = Loader()
    e = load('elements')
    assert 118 == e.shape[0], 'should have 118 elements'
    assert 74 == e.shape[1], 'should have 74 features'

    e = load('elements_completed')
    assert 94 == e.shape[0], 'should have 94 completed elements'
    assert 58 == e.shape[1], 'should have 58 completed features'


def test_loader_url1(setup):
    load = Loader(setup['fetch_file'])
    file = load()
    with open(str(file)) as f:
        assert f.readline() == 'Test xenonpy.utils.Loader._fetch_data'
    remove(str(file))


def test_loader_url2(setup):
    load = Loader(setup['fetch_file'])
    file = load('fetch_test.txt')
    with open(str(file)) as f:
        assert f.readline() == 'Test xenonpy.utils.Loader._fetch_data'
    remove(str(file))


def test_loader_property():
    load = Loader()
    assert 118 == load.elements.shape[0], 'should have 118 elements'
    assert 74 == load.elements.shape[1], 'should have 74 features'

    assert 94 == load.elements_completed.shape[
        0], 'should have 94 completed elements'
    assert 58 == load.elements_completed.shape[
        1], 'should have 58 completed features'


def test_loader_return_dataset(setup):
    load = Loader()
    dir_ = setup['test_dir']
    try:
        load(dir_)
    except FileNotFoundError:
        assert True
        return

    assert False, 'should got FileNotFoundError'


# =====================================================================================================
def test_dataset5(setup):
    saver = DataSet('dens')
    saver(list('sdfs'))


def test_dataset1(setup):
    saver = DataSet(setup['user_dataset'])
    ret = '<{}> include:'.format(setup['user_dataset'])
    assert str(saver) == ret, 'no files'


def test_dataset2(setup):
    saver = DataSet(setup['user_dataset'])
    saver(list('abcd'), list('efgh'))
    assert len(saver._files['unnamed']) == 2, 'should got 2 files'


def test_dataset3(setup):
    saver = DataSet(setup['user_dataset'])
    saver(key1=list('asdf'), key2=list('qwer'))
    assert len(saver._files['unnamed']) == 2, 'should got 2 files'
    assert len(saver._files['key1']) == 1, 'should got 1 files'
    assert len(saver._files['key2']) == 1, 'should got 1 files'


def test_dataset4(setup):
    saver = DataSet(setup['user_dataset'])
    saver(list('asdf'), key1=list('qwer'))
    assert len(saver._files['unnamed']) == 3, 'should got 3 files'
    assert len(saver._files['key1']) == 2, 'should got 1 files'
    assert len(saver._files['key2']) == 1, 'should got 1 files'


def test_dataset_prop(setup):
    saver = DataSet(setup['user_dataset'])
    assert saver.name == 'test_user_data'
    assert saver.path == str(Path.home() / '.xenonpy' / 'userdata')


def test_dataset_last1(setup):
    saver = DataSet(setup['user_dataset'])
    last = saver.last()
    assert last == list('asdf'), 'retriever same data'


def test_dataset_last2(setup):
    saver = DataSet(setup['user_dataset'])
    last = saver.last('key1')
    assert last == list('qwer'), 'retriever same data'


def test_dataset_getitem1(setup):
    saver = DataSet(setup['user_dataset'])
    item = saver[:]
    assert item[1] == list('efgh'), 'retriever same data'
    item = saver[1]
    assert item == list('efgh'), 'retriever same data'


def test_dataset_getitem2(setup):
    saver = DataSet(setup['user_dataset'])
    item = saver['key2', :]
    assert item[0] == list('qwer'), 'retriever same data'
    item = saver['key1', 1]
    assert item == list('qwer'), 'retriever same data'


def test_dump1(setup):
    saver = DataSet(setup['user_dataset'])
    path = Path().expanduser()
    path_ = saver.dump(path, with_datetime=False)
    assert path_ == str(path / (setup['user_dataset'] + '.pkl.z'))


def test_dump2(setup):
    path = Path().expanduser()
    path = path / (setup['user_dataset'] + '.pkl.z')
    dumped = jl.load(path)
    assert dumped['key1'] == list('qwer')


def test_dataset_chained(setup):
    saver = DataSet(setup['user_dataset']).chain
    assert saver.name == 'chain'
    assert saver.path == str(Path.home() / '.xenonpy' / 'userdata' / 'test_user_data')


def test_dataset_delete1(setup):
    saver = DataSet(setup['user_dataset'])
    saver.rm(0)
    assert len(saver._files['unnamed']) == 2, 'should got 1 files'


def test_dataset_delete2(setup):
    saver = DataSet(setup['user_dataset'])
    saver(key1=list('qwer'))
    assert len(saver._files['key1']) == 3, 'should got 3 files'
    saver.rm(slice(0, 2), 'key1')
    assert len(saver._files['key1']) == 1, 'should got 1 files'


def test_dataset_clean1(setup):
    saver = DataSet(setup['user_dataset'])
    saver.clean('key1')
    assert 'key1' not in saver._files, 'no saver dir'


def test_dataset_clean2(setup):
    saver = DataSet(setup['user_dataset'])
    saver_dir = Path.home() / '.xenonpy' / 'userdata' / setup['user_dataset']
    saver.clean()
    assert not saver_dir.exists(), 'no saver dir'


if __name__ == '__main__':
    pytest.main()
