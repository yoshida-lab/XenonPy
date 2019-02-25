#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import pytest

from xenonpy.datatools import MDL


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield MDL()

    print('test over')


def test_query_properties(data):
    mdl = data
    ret = mdl.query_properties('test')
    assert isinstance(ret, list)


def test_query_models(data):
    mdl = data
    ret = mdl._query_models('test')
    assert isinstance(ret, list)


def test_fetch_models(data):
    mdl = data
    # ret = mdl('test', '.')
    # assert isinstance(ret, pd.DataFrame)

    # rmtree(str(Path.home() / '.xenonpy/userdata' / test['test_dir']))

    ret = mdl('some_thing_not_exist')
    assert ret is None


if __name__ == "__main__":
    pytest.main()
