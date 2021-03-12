#  Copyright (c) 2021. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from xenonpy.descriptor import ECFP, FCFP, AtomPairFP, RDKitFP, MACCS, Fingerprints


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


def test_ecfp_1(data):
    fps = ECFP(n_jobs=1)
    fps.transform(data['mols'])

    with pytest.raises(TypeError):
        fps.transform(data['smis'])


def test_ecfp_2(data):
    fps = ECFP(n_jobs=1, input_type='smiles')
    with pytest.raises(TypeError):
        fps.transform(data['mols'])

    fps.transform(data['smis'])


def test_ecfp_3(data):
    fps = ECFP(n_jobs=1, input_type='any')
    fps.transform(data['mols'] + data['smis'])


def test_ecfp_4(data):
    fps = ECFP(n_jobs=1, input_type='any')
    with pytest.raises(ValueError):
        fps.transform(data['err_smis'])

    fps = ECFP(n_jobs=1, input_type='any', on_errors='nan')
    ret = fps.transform(data['err_smis'])
    assert pd.DataFrame(data=ret).shape == (4, 2048)
    assert np.isnan(ret[1][10])
    assert np.isnan(ret[2][20])


def test_fcfp_1(data):
    fps = FCFP(n_jobs=1)
    fps.transform(data['mols'])

    with pytest.raises(TypeError):
        fps.transform(data['smis'])


def test_fcfp_2(data):
    fps = FCFP(n_jobs=1, input_type='smiles')
    with pytest.raises(TypeError):
        fps.transform(data['mols'])

    fps.transform(data['smis'])


def test_fcfp_3(data):
    fps = FCFP(n_jobs=1, input_type='any')
    fps.transform(data['mols'] + data['smis'])


def test_fcfp_4(data):
    fps = FCFP(n_jobs=1, input_type='any')
    with pytest.raises(ValueError):
        fps.transform(data['err_smis'])

    fps = FCFP(n_jobs=1, input_type='any', on_errors='nan')
    ret = fps.transform(data['err_smis'])
    assert pd.DataFrame(data=ret).shape == (4, 2048)
    assert np.isnan(ret[1][10])
    assert np.isnan(ret[2][20])


def test_apfp_1(data):
    fps = AtomPairFP(n_jobs=1)
    fps.transform(data['mols'])

    with pytest.raises(TypeError):
        fps.transform(data['smis'])


def test_apfp_2(data):
    fps = AtomPairFP(n_jobs=1, input_type='smiles')
    with pytest.raises(TypeError):
        fps.transform(data['mols'])

    fps.transform(data['smis'])


def test_apfp_3(data):
    fps = AtomPairFP(n_jobs=1, input_type='any')
    fps.transform(data['mols'] + data['smis'])


def test_apfp_4(data):
    fps = AtomPairFP(n_jobs=1, input_type='any')
    with pytest.raises(ValueError):
        fps.transform(data['err_smis'])

    fps = AtomPairFP(n_jobs=1, input_type='any', on_errors='nan')
    ret = fps.transform(data['err_smis'])
    assert pd.DataFrame(data=ret).shape == (4, 2048)
    assert np.isnan(ret[1][10])
    assert np.isnan(ret[2][20])


def test_rdfp_1(data):
    fps = RDKitFP(n_jobs=1)
    fps.transform(data['mols'])

    with pytest.raises(TypeError):
        fps.transform(data['smis'])


def test_rdfp_2(data):
    fps = RDKitFP(n_jobs=1, input_type='smiles')
    with pytest.raises(TypeError):
        fps.transform(data['mols'])

    fps.transform(data['smis'])


def test_rdfp_3(data):
    fps = RDKitFP(n_jobs=1, input_type='any')
    fps.transform(data['mols'] + data['smis'])


def test_rdfp_4(data):
    fps = RDKitFP(n_jobs=1, input_type='any')
    with pytest.raises(ValueError):
        fps.transform(data['err_smis'])

    fps = RDKitFP(n_jobs=1, input_type='any', on_errors='nan')
    ret = fps.transform(data['err_smis'])
    assert pd.DataFrame(data=ret).shape == (4, 2048)
    assert np.isnan(ret[1][10])
    assert np.isnan(ret[2][20])


def test_maccs_1(data):
    fps = MACCS(n_jobs=1)
    fps.transform(data['mols'])

    with pytest.raises(TypeError):
        fps.transform(data['smis'])


def test_maccs_2(data):
    fps = MACCS(n_jobs=1, input_type='smiles')
    with pytest.raises(TypeError):
        fps.transform(data['mols'])

    fps.transform(data['smis'])


def test_maccs_3(data):
    fps = MACCS(n_jobs=1, input_type='any')
    fps.transform(data['mols'] + data['smis'])


def test_maccs4(data):
    fps = MACCS(n_jobs=1, input_type='any')
    with pytest.raises(ValueError):
        fps.transform(data['err_smis'])

    fps = MACCS(n_jobs=1, input_type='any', on_errors='nan')
    ret = fps.transform(data['err_smis'])
    assert pd.DataFrame(data=ret).shape == (4, 167)
    assert np.isnan(ret[1][10])
    assert np.isnan(ret[2][20])


def test_fps_1(data):
    fps = Fingerprints(n_jobs=1)
    fps.transform(data['mols'])

    with pytest.raises(TypeError):
        fps.transform(data['smis'])


def test_fps_2(data):
    fps = Fingerprints(n_jobs=1, input_type='smiles')
    with pytest.raises(TypeError):
        fps.transform(data['mols'])

    fps.transform(data['smis'])


def test_fps_3(data):
    fps = Fingerprints(n_jobs=1, input_type='any')
    fps.transform(data['mols'] + data['smis'])


def test_fps_4(data):
    fps = Fingerprints(n_jobs=1, input_type='any')
    with pytest.raises(ValueError):
        fps.transform(data['err_smis'])

    fps = Fingerprints(n_jobs=1, input_type='any', on_errors='nan')
    ret = fps.transform(data['err_smis'])
    assert isinstance(ret, pd.DataFrame)
    assert ret.shape == (4, 16759)
    assert np.isnan(ret.values[1][10])
    assert np.isnan(ret.values[2][20])


def test_fps_5(data):
    fps = Fingerprints(featurizers='ECFP', n_jobs=1, input_type='any', on_errors='nan', n_bits=100)
    ret = fps.transform(data['err_smis'])
    assert isinstance(ret, pd.DataFrame)
    assert ret.shape == (4, 100)
    assert np.isnan(ret.values[1][10])
    assert np.isnan(ret.values[2][20])


def test_fps_6(data):
    fps = Fingerprints(n_jobs=1, input_type='any', on_errors='nan', counting=True)
    ret = fps.transform(data['smis'])
    assert isinstance(ret, pd.DataFrame)
    assert ret.shape == (4, 16759)


if __name__ == "__main__":
    pytest.main()
