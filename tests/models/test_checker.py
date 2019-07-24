#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from copy import deepcopy
from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest

from xenonpy.datatools import Storage
from xenonpy.model.layer import Layer1d
from xenonpy.model.utils import Checker


@pytest.fixture(scope='module')
def setup():
    # prepare path
    name = 'test_checker'
    default = Path.home() / '.xenonpy/usermodel'
    dot = Path()
    test = dict(
        name=name,
        cp=dict(model_state=1, b=2),
        path='./',
        default=str(default),
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

    dirs_ = [d for d in default.iterdir() if d.match(name + '@*')] + \
            [d for d in dot.iterdir() if d.match(name + '@*')]
    for d in dirs_:
        rmtree(str(d))
    print('test over')


def test_checker_omit_path(setup):
    checker = Checker(setup['name'])
    assert isinstance(checker, Storage)
    assert checker.path == setup['default']
    assert checker.name == setup['name'] + '@1'


def test_checker_default_path(setup):
    checker = Checker(setup['name'], path=setup['default'])
    assert isinstance(checker, Storage)
    assert checker.path == setup['default']
    assert checker.name == setup['name'] + '@2'


def test_checker_assign_path(setup):
    checker = Checker(setup['name'], path=setup['dot'])
    assert isinstance(checker, Storage)
    assert checker.path == str(Path().absolute())
    assert checker.name == setup['name'] + '@1'


def test_checker_init_model_1(setup):
    checker = Checker(setup['name'])
    with pytest.raises(TypeError):
        checker.init_model = None


def test_checker_init_model_2(setup):
    checker = Checker(setup['name'])
    checker.init_model = setup['model']


def test_checker_init_model_3(setup):
    checker = Checker(setup['name'])
    checker.init_model = setup['model']
    assert str(checker.init_model) == str(setup['model'])


def test_checker_trained_model_1(setup):
    checker = Checker(setup['name'])
    with pytest.raises(TypeError):
        checker.trained_model = None


def test_checker_trained_model_2(setup):
    checker = Checker(setup['name'])
    checker.trained_model = setup['model']


def test_checker_trained_model_3(setup):
    checker = Checker(setup['name'])
    checker.trained_model = setup['model']
    assert str(checker.trained_model) == str(setup['model'])


def test_checker_call(setup):
    checker = Checker(setup['name'])
    checker(**setup['cp'])
    assert (Path(checker.path) / checker.name / 'checkpoints').exists()


def test_checker_from_cp(setup):
    checker = Checker(setup['name'])
    name = checker.name
    path = checker.path
    path = Path(path) / name
    checker(**setup['cp'])
    checker2 = Checker.load(str(path))
    model_state, other = checker2[0]
    other_ = deepcopy(setup['cp'])
    del other_['model_state']
    assert model_state == setup['cp']['model_state']
    assert other == other_


if __name__ == "__main__":
    pytest.main()
