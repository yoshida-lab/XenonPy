# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from pathlib import Path
from shutil import rmtree

import numpy as np
import pandas as pd
import pytest

from xenonpy.model.nn import Checker, Layer1d
from xenonpy.utils import DataSet


@pytest.fixture(scope='module')
def setup():
    # prepare path
    name = 'test'
    default = Path.home() / '.xenonpy/usermodel'
    dot = Path()
    test = dict(
        name=name,
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

    dirs_ = [d for d in default.iterdir() if d.match('test@*')] + [d for d in dot.iterdir() if d.match('test@*')]
    for d in dirs_:
        rmtree(str(d))
    print('test over')


def test_checker_omit_path(setup):
    checker = Checker(setup['name'])
    assert isinstance(checker, DataSet)
    assert checker.path == setup['default']
    assert checker.name == setup['name'] + '@1'


def test_checker_default_path(setup):
    checker = Checker(setup['name'], path=setup['default'])
    assert isinstance(checker, DataSet)
    assert checker.path == setup['default']
    assert checker.name == setup['name'] + '@2'


def test_checker_assign_path(setup):
    checker = Checker(setup['name'], path=setup['dot'])
    assert isinstance(checker, DataSet)
    assert checker.path == str(Path().resolve())
    assert checker.name == setup['name'] + '@1'


def test_checker_init_model1(setup):
    checker = Checker(setup['name'])
    try:
        checker.init_model = None
    except TypeError:
        assert True
    assert False, 'should got error'


def test_checker_init_model2(setup):
    checker = Checker(setup['name'])
    try:
        checker.init_model = setup['model']
    except TypeError:
        assert False, 'should not got error'


def test_checker_trained_model1(setup):
    checker = Checker(setup['name'])
    try:
        checker.trained_model = None
    except TypeError:
        assert True
    assert False, 'should got error'


def test_checker_trained_model2(setup):
    checker = Checker(setup['name'])
    try:
        checker.trained_model = setup['model']
    except TypeError:
        assert False, 'should not got error'


def test_checker_train_data1(setup):
    checker = Checker(setup['name'])
    try:
        checker.train_data(None, None)
    except TypeError:
        assert True
    assert False, 'should got error'


def test_checker_train_data2(setup):
    checker = Checker(setup['name'])
    try:
        checker.train_data(setup['np'], setup['df'])
    except TypeError:
        assert False, 'should not got error'


if __name__ == "__main__":
    pytest.main()
