#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import os
from pathlib import Path

import pytest

from xenonpy.utils import set_env, absolute_path, config


def test_set_env_1():
    import os
    with set_env(test_env='test'):
        assert os.getenv('test_env') == 'test', 'should got "test"'
    assert os.getenv('test_env') is None, 'no test_env'


def test_set_env_2():
    import os
    os.environ['test_env'] = 'foo'
    with set_env(test_env='bar'):
        assert os.getenv('test_env') == 'bar', 'should got "bar"'
    assert os.getenv('foo') is None, 'foo'


def test_absolute_path_1():
    path = absolute_path('.')
    path = Path(path)
    cwd = Path(os.getcwd())
    assert path.parent.name == cwd.parent.name
    assert path.name == cwd.name


def test_absolute_path_2():
    path = absolute_path('../')
    path = Path(path)
    cwd = Path(os.getcwd())
    assert path.name == cwd.parent.name


def test_absolute_path_3():
    path = absolute_path('../other')
    path = Path(path)
    cwd = Path(os.getcwd())
    assert path.name == 'other'
    assert path.parent.name == cwd.parent.name


def test_config_1():
    tmp = config('name')
    assert tmp == 'xenonpy'

    with pytest.raises(RuntimeError):
        config('no_exist')

    tmp = config(new_key='test')
    assert tmp is None

    tmp = config('new_key')
    assert tmp == 'test'

    tmp = config('version', other_key='other')
    assert tmp == '0.2.3'

    tmp = config('other_key')
    assert tmp == 'other'


if __name__ == "__main__":
    pytest.main()
