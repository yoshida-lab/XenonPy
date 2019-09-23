#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pytest

from xenonpy.utils import ParameterGenerator


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    yield dict(
        int=5,
        tuple=(0, 1, 2, 3),
        func=lambda: np.random.randint(4),
        tuple_dict_1=dict(data=(0, 1, 2, 3), repeat=(1, 2, 3)),
        func_dict_1=dict(data=lambda i: np.random.randint(4, size=i), repeat=(1, 2, 3)),
        tuple_dict_2=dict(data=(0, 1, 2, 3), repeat=2),
        func_dict_2=dict(data=lambda i: np.random.randint(4, size=i), repeat='tuple_dict_1'),
    )
    print('test over')


def test_gen_1():
    with pytest.raises(RuntimeError, match='need parameter candidate'):
        ParameterGenerator()

    test_data = (1, 2, 3)
    pg = ParameterGenerator(a=test_data)
    i_ = 0
    for i, t in enumerate(pg(5)):
        assert i_ == i
        i_ += 1

    for t, p in pg(5, factory=lambda a: a + 1):
        assert t['a'] + 1 == p


def test_gen_2(data):
    pg = ParameterGenerator(name=data['tuple'])
    for t in pg(10):
        assert 'name' in t
        assert t['name'] in data['tuple']
        assert 0 <= t['name'] < 4


def test_gen_3(data):
    pg = ParameterGenerator(name=data['func'])
    for t in pg(10):
        assert 0 <= t['name'] < 4


def test_gen_4(data):
    pg = ParameterGenerator(name=data['tuple_dict_1'])
    for t in pg(10):
        assert isinstance(t['name'], tuple)
        assert len(t['name']) <= 3
        for i in t['name']:
            assert 0 <= i <= 3


def test_gen_5(data):
    pg = ParameterGenerator(name=data['func_dict_1'])
    for t in pg(10):
        assert len(t['name']) <= 3
        for i in t['name']:
            assert 0 <= i <= 3


def test_gen_6(data):
    pg = ParameterGenerator(name=data['tuple_dict_2'])
    for t in pg(10):
        assert len(t['name']) == 2
        for i in t['name']:
            assert 0 <= i <= 3


def test_gen_7(data):
    pg = ParameterGenerator(name=data['int'])
    for t in pg(10):
        assert t['name'] == 5


def test_gen_8(data):
    pg = ParameterGenerator(name=data['func_dict_2'])
    with pytest.raises(KeyError):
        for _ in pg(10):
            pass

    pg = ParameterGenerator(name=data['func_dict_2'], tuple_dict_1=data['tuple_dict_1'])
    for t in pg(10):
        assert 'tuple_dict_1' in t
        assert len(t['name']) == len(t['tuple_dict_1'])
        for i in t['name']:
            assert 0 <= i <= 3
        for i in t['name']:
            assert 0 <= i <= 3

    pg = ParameterGenerator(tuple_dict_1=data['tuple_dict_1'], name=data['func_dict_2'])
    for t in pg(10):
        assert 'tuple_dict_1' in t
        assert len(t['name']) == len(t['tuple_dict_1'])
        for i in t['name']:
            assert 0 <= i <= 3
        for i in t['name']:
            assert 0 <= i <= 3


if __name__ == "__main__":
    pytest.main()
