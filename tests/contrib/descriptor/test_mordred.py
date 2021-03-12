#  Copyright (c) 2021. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pandas as pd
import pytest
from mordred._base.pandas_module import MordredDataFrame
from rdkit import Chem

from xenonpy.contrib.extend_descriptors.descriptor import Mordred2DDescriptor


@pytest.fixture(scope='module')
def data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    smis = ['C(C(O)C1(O))C(CO)OC1O',
            'CC(C1=CC=CC=C1)CC(C2=CC=CC=C2)CC(C3=CC=CC=C3)CC(C4=CC=CC=C4)',
            ' CC(C)CC(C)CC(C)',
            'C(F)C(F)(F)']

    mols = [Chem.MolFromSmiles(s) for s in smis]

    err_smis = ['C(C(O)C1(O))C(CO)OC1O',
                'CC(C1=CC=CC=C1)CC(C2=CC=CC=C2)CC(C3=CC=',
                'Ccccccc',
                'C(F)C(F)(F)']
    yield dict(smis=smis, mols=mols, err_smis=err_smis)

    print('test over')


def test_mordred_1(data):
    mordred = Mordred2DDescriptor()
    desc = mordred.transform(data['smis'])
    assert isinstance(desc, MordredDataFrame)

    mordred = Mordred2DDescriptor(return_type='df')
    desc = mordred.transform(data['smis'])
    assert isinstance(desc, pd.DataFrame)


def test_mordred_2(data):
    mordred = Mordred2DDescriptor()
    desc = mordred.transform(data['mols'])
    assert isinstance(desc, MordredDataFrame)


def test_mordred_3(data):
    mordred = Mordred2DDescriptor()
    with pytest.raises(ValueError):
        mordred.transform(data['err_smis'])


if __name__ == "__main__":
    pytest.main()
