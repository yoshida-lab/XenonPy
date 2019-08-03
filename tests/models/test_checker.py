#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import os
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest

from xenonpy.model.nn import Layer1d
from xenonpy.model.utils import Checker


@pytest.fixture(scope='module')
def setup():
    # prepare path
    name = 'checker'
    dir_ = os.path.dirname(os.path.abspath(__file__))
    dot = Path()
    test = dict(
        name=name,
        dir_=dir_,
        cp=dict(model_state=1, b=2),
        path=f'{dir_}/test/{name}',
        dot=str(dot),
        model=Layer1d(10, 1),
        np=np.ones((2, 3)),
        df=pd.DataFrame(np.ones((2, 3)))
    )

    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield test

    rmtree(f'{dir_}/test')

    print('test over')


def test_checker_path(setup):
    checker = Checker(setup['path'])
    assert checker.path == str(Path(setup['path']))


def test_checker_default_path(setup):
    checker = Checker(setup['path'])
    assert checker.path == str(Path(setup['path']))

    checker = Checker(setup['path'], increment=True)
    assert checker.path == str(Path(setup['path'] + '@1'))


def test_checker_init_model_1(setup):
    checker = Checker(setup['path'])
    with pytest.raises(TypeError):
        checker.init_model = None

    checker.init_model = setup['model']
    assert isinstance(checker.init_model, Layer1d)
    assert str(checker.init_model) == str(setup['model'])


def test_checker_trained_model_1(setup):
    checker = Checker(setup['path'])
    with pytest.raises(TypeError):
        checker.trained_model = None

    checker.trained_model = setup['model']
    assert isinstance(checker.trained_model, Layer1d)
    assert str(checker.trained_model) == str(setup['model'])


def test_checker_call(setup):
    checker = Checker(setup['path'])
    checker.set_checkpoint(**setup['cp'])
    assert (Path(checker.path) / 'checkpoints').exists()


def test_checker_from_cp(setup):
    checker = Checker(setup['path'])
    path = checker.path
    checker.set_checkpoint(test_cp=setup['cp'])
    checker(a=1)
    checker2 = Checker.load(path)
    assert checker2['a'] == 1

    cp = checker2.checkpoints['test_cp']
    assert 'b' in cp
    assert cp['model_state'] == setup['cp']['model_state']


if __name__ == "__main__":
    pytest.main()
