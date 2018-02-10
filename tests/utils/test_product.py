# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


from itertools import product

import pytest

from xenonpy.utils import Product


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    # prepare test data
    a = ['a1', 'a2', 'a3']
    b = ['b1', 'b2']
    c = ['c1', 'c2']
    d = ['d1', 'd2', 'd3']
    yield (a, b, c, d)

    print('test over')


def test_product1(data):
    abcd = list(product(*data))
    p = Product(*data)
    assert p.size == len(abcd), 'should have same length'
    assert p[0] == abcd[0]
    assert p[1] == abcd[1]
    assert p[3] != abcd[4]
    assert p[10] == abcd[10]
    assert p[22] == abcd[22]
    assert p[33] == abcd[33]

    try:
        p[36]
    except IndexError:
        assert True
    else:
        assert False, 'should got IndexError'


def test_product2(data):
    try:
        Product(data[0], repeat=1.5)
    except ValueError:
        assert True
    else:
        assert False, 'should got ValueError'


def test_product3(data):
    abcd = list(product(data[0], repeat=1))
    p = Product(data[0], repeat=1)
    assert p.size == len(abcd), 'should have same length'
    assert p[0] == abcd[0]
    assert p[1] == abcd[1]
    assert p[2] == abcd[2]


def test_product4(data):
    abcd = list(product(data[0], repeat=3))
    p = Product(data[0], repeat=3)
    assert p.size == len(abcd), 'should have same length'
    assert p[0] == abcd[0]
    assert p[1] == abcd[1]
    assert p[3] != abcd[4]
    assert p[5] == abcd[5]
    print(p.paras)


def test_product5(data):
    abcd = list(product(product(*data), repeat=2))
    p = Product(Product(*data), repeat=2)
    assert p.size == len(abcd), 'should have same length'
    assert p[0] == abcd[0]
    assert p[1] == abcd[1]
    assert p[50] != abcd[55]
    assert p[333] == abcd[333]
    assert p[555] == abcd[555]
    assert p[666] != abcd[777]


if __name__ == "__main__":
    pytest.main()
