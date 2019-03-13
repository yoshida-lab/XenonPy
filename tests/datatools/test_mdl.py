#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


from json import JSONDecodeError

import pytest
from requests import HTTPError

from xenonpy.datatools import MDL


@pytest.fixture(scope='module')
def mdl():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield MDL()

    print('test over')


def test_query_properties(mdl):
    ret = mdl.query_properties('test')
    assert isinstance(ret, list)


def test_query_models(mdl):
    ret = mdl._query_models('test')
    assert isinstance(ret, list)


def test_fetch_models(mdl):
    # ret = mdl('test', '.')
    # assert isinstance(ret, pd.DataFrame)

    # rmtree(str(Path.home() / '.xenonpy/userdata' / test['test_dir']))

    ret = mdl('some_thing_not_exist')
    assert ret is None


def test_return_nothing(mdl, monkeypatch):
    class Request_Dummy(object):
        def __init__(self, **_):
            self.status_code = 999

        def json(self):
            raise JSONDecodeError("error", "error", 0)

    monkeypatch.setattr("requests.post", Request_Dummy)
    with pytest.raises(HTTPError) as excinfo:
        mdl("test", save_to=False)
    exc_msg = excinfo.value.args[0]
    assert exc_msg == "status_code: 999, Server did not responce."


if __name__ == "__main__":
    pytest.main()
