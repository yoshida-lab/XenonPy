#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import os
from pathlib import Path

from xenonpy.utils import set_env, absolute_path


def test_set_env1():
    import os
    with set_env(test_env='test'):
        assert os.getenv('test_env') == 'test', 'should got "test"'
    assert os.getenv('test_env') is None, 'no test_env'


def test_set_env2():
    import os
    os.environ['test_env'] = 'foo'
    with set_env(test_env='bar'):
        assert os.getenv('test_env') == 'bar', 'should got "bar"'
    assert os.getenv('foo') is None, 'foo'


def test_absolute_path1():
    path = absolute_path('.')
    path = Path(path)
    cwd = Path(os.getcwd())
    assert path.parent.name == cwd.parent.name
    assert path.name == cwd.name


def test_absolute_path2():
    path = absolute_path('../')
    path = Path(path)
    cwd = Path(os.getcwd())
    assert path.name == cwd.parent.name


def test_absolute_path3():
    path = absolute_path('../other')
    path = Path(path)
    cwd = Path(os.getcwd())
    assert path.name == 'other'
    assert path.parent.name == cwd.parent.name
