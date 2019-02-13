#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from pathlib import Path

import pandas as pd
import pytest
from pymatgen import Structure as pmg_S

from xenonpy.descriptor import RadialDistributionFunction, Structures, ObitalFieldMatrix


@pytest.fixture(scope='module')
def data():
    # prepare path
    pwd = Path(__file__).parent
    cif1 = pmg_S.from_file(str(pwd / '1.cif'))
    cif2 = pmg_S.from_file(str(pwd / '2.cif'))

    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    cifs = pd.Series([cif1, cif2], name='structure')
    yield cifs

    print('test over')


def test_rdf(data):
    RadialDistributionFunction().fit_transform(data)
    assert True


def test_ofm(data):
    ObitalFieldMatrix().fit_transform(data)
    assert True


def test_structure(data):
    Structures().fit_transform(data)
    assert True


if __name__ == "__main__":
    pytest.main()
