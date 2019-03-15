#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from os import remove
from pathlib import Path
from shutil import rmtree

import pytest

from xenonpy import __cfg_root__
from xenonpy.datatools import Preset, Storage, preset


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


def test_preset_1():
    assert Preset() is preset
    e = preset.elements
    assert 118 == e.shape[0], 'should have 118 elements'
    assert 74 == e.shape[1], 'should have 74 features'

    e = preset.elements_completed
    assert 94 == e.shape[0], 'should have 94 completed elements'
    assert 58 == e.shape[1], 'should have 58 completed features'


def test_preset_2():
    assert hasattr(preset, 'dataset_elements')
    e = preset.dataset_elements
    assert 118 == e.shape[0], 'should have 118 elements'
    assert 74 == e.shape[1], 'should have 74 features'

    assert hasattr(preset, 'dataset_elements_completed')
    e = preset.dataset_elements_completed
    assert 94 == e.shape[0], 'should have 94 completed elements'
    assert 58 == e.shape[1], 'should have 58 completed features'


def test_preset_3():
    path = Path(__cfg_root__) / 'dataset' / 'elements.pkl.pd_'
    if path.exists():
        remove(str(path))

    with pytest.raises(RuntimeError, match="data elements not exist"):
        preset.elements

    preset.sync('elements')
    preset.elements


if __name__ == '__main__':
    pytest.main()
