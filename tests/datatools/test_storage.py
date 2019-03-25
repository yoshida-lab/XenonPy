#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from os import remove
from pathlib import Path
from shutil import rmtree

import pytest
from sklearn.externals import joblib as jl

from xenonpy.datatools import Storage


@pytest.fixture(scope='module')
def setup():
    # prepare path
    test = dict(
        test_dir='test_dir',
        user_dataset='test_user_data',
        saver=Storage('test_user_data'))

    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield test

    if (Path.home() / '.xenonpy/userdata' / test['test_dir']).exists():
        rmtree(str(Path.home() / '.xenonpy/userdata' / test['test_dir']))
    if (Path.home() / '.xenonpy/cached/travis').exists():
        rmtree(str(Path.home() / '.xenonpy/cached/travis'))
    if (Path().expanduser() / 'test_user_data.pkl.z').exists():
        remove(str(Path().expanduser() / 'test_user_data.pkl.z'))
    print('test over')


def test_storage_1(setup):
    saver = setup['saver']
    ret = '<{}> under `{}` includes:'.format(setup['user_dataset'],
                                             Path.home() / '.xenonpy' / 'userdata')
    assert str(saver) == ret, 'no files'


def test_storage_2(setup):
    saver = setup['saver']
    saver(list('abcd'), list('efgh'))
    assert len(saver._files['unnamed']) == 2, 'should got 2 files'


def test_storage_3(setup):
    saver = setup['saver']
    saver(key1=list('asdf'), key2=list('qwer'))
    assert len(saver._files['unnamed']) == 2, 'should got 2 files'
    assert len(saver._files['key1']) == 1, 'should got 1 files'
    assert len(saver._files['key2']) == 1, 'should got 1 files'


def test_storage_4(setup):
    saver = setup['saver']
    saver(list('asdf'), key1=list('qwer'))
    assert len(saver._files['unnamed']) == 3, 'should got 3 files'
    assert len(saver._files['key1']) == 2, 'should got 1 files'
    assert len(saver._files['key2']) == 1, 'should got 1 files'


def test_storage_prop(setup):
    saver = setup['saver']
    assert saver.name == 'test_user_data'
    assert saver.path == str(Path.home() / '.xenonpy' / 'userdata')


def test_storage_last_1(setup):
    saver = setup['saver']
    last = saver.last()
    assert last == list('asdf'), 'retriever same data'


def test_storage_last_2(setup):
    saver = setup['saver']
    last = saver.last('key1')
    assert last == list('qwer'), 'retriever same data'


def test_storage_getitem_1(setup):
    saver = setup['saver']
    item = saver[:]
    assert item[1] == list('efgh'), 'retriever same data'
    item = saver[1]
    assert item == list('efgh'), 'retriever same data'


def test_storage_getitem_2(setup):
    saver = setup['saver']
    item = saver['key2', :]
    assert item[0] == list('qwer'), 'retriever same data'
    item = saver['key1', 1]
    assert item == list('qwer'), 'retriever same data'


def test_dump_1(setup):
    saver = setup['saver']
    path = Path().expanduser()
    path_ = saver.dump(path, with_datetime=False)
    assert path_ == str(path / (setup['user_dataset'] + '.pkl.z'))


def test_dump_2(setup):
    path = Path().expanduser()
    path = path / (setup['user_dataset'] + '.pkl.z')
    dumped = jl.load(path)
    assert dumped['key1'] == list('qwer')


def test_storage_chained(setup):
    saver = setup['saver'].chain
    assert saver.name == 'chain'
    assert saver.path == str(Path.home() / '.xenonpy' / 'userdata' / 'test_user_data')


def test_storage_delete_1(setup):
    saver = setup['saver']
    saver.rm(0)
    assert len(saver._files['unnamed']) == 2, 'should got 1 files'


def test_storage_delete_2(setup):
    saver = setup['saver']
    saver(key1=list('qwer'))
    assert len(saver._files['key1']) == 3, 'should got 3 files'
    saver.rm(slice(0, 2), 'key1')
    assert len(saver._files['key1']) == 1, 'should got 1 files'


def test_storage_clean_1(setup):
    saver = setup['saver']
    saver.clean('key1')
    assert 'key1' not in saver._files, 'no saver dir'


def test_storage_clean_2(setup):
    saver = setup['saver']
    saver_dir = Path.home() / '.xenonpy' / 'userdata' / setup['user_dataset']
    saver.clean()
    assert not saver_dir.exists(), 'no saver dir'


if __name__ == '__main__':
    pytest.main()
