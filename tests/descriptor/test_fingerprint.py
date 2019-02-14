#  Copyright (c) 2019. yoshida-lab. All rights reserved.
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

    try:
        fps.transform(data['mols'])
    except:
        assert False, 'should not got error'
    else:
        assert True

    try:
        fps.transform(data['smis'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'


def test_ecfp_2(data):
    fps = ECFP(n_jobs=1, input_type='smiles')
    try:
        fps.transform(data['mols'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'

    try:
        fps.transform(data['smis'])
    except:
        assert False, 'should not got error'
    else:
        assert True


def test_ecfp_3(data):
    fps = ECFP(n_jobs=1, input_type='any')
    try:
        fps.transform(data['mols'] + data['smis'])
    except BaseException as e:
        print(e)
        assert False, 'should not got error'
    else:
        assert True


def test_ecfp_4(data):
    fps = ECFP(n_jobs=1, input_type='any')
    try:
        fps.transform(data['err_smis'])
    except ValueError:
        assert True
    else:
        assert False, 'should not got error'

    fps = ECFP(n_jobs=1, input_type='any', on_errors='nan')
    try:
        ret = fps.transform(data['err_smis'])
        assert pd.DataFrame(data=ret).shape == (4, 2048)
        assert np.isnan(ret[1][10])
        assert np.isnan(ret[2][20])
    except:
        assert False
    else:
        assert True


def test_fcfp_1(data):
    fps = FCFP(n_jobs=1)

    try:
        fps.transform(data['mols'])
    except:
        assert False, 'should not got error'
    else:
        assert True

    try:
        fps.transform(data['smis'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'


def test_fcfp_2(data):
    fps = FCFP(n_jobs=1, input_type='smiles')
    try:
        fps.transform(data['mols'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'

    try:
        fps.transform(data['smis'])
    except:
        assert False, 'should not got error'
    else:
        assert True


def test_fcfp_3(data):
    fps = FCFP(n_jobs=1, input_type='any')
    try:
        fps.transform(data['mols'] + data['smis'])
    except BaseException as e:
        print(e)
        assert False, 'should not got error'
    else:
        assert True


def test_fcfp_4(data):
    fps = FCFP(n_jobs=1, input_type='any')
    try:
        fps.transform(data['err_smis'])
    except ValueError:
        assert True
    else:
        assert False, 'should not got error'

    fps = ECFP(n_jobs=1, input_type='any', on_errors='nan')
    try:
        ret = fps.transform(data['err_smis'])
        assert pd.DataFrame(data=ret).shape == (4, 2048)
        assert np.isnan(ret[1][10])
        assert np.isnan(ret[2][20])
    except:
        assert False
    else:
        assert True


def test_apfp_1(data):
    fps = AtomPairFP(n_jobs=1)

    try:
        fps.transform(data['mols'])
    except:
        assert False, 'should not got error'
    else:
        assert True

    try:
        fps.transform(data['smis'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'


def test_apfp_2(data):
    fps = AtomPairFP(n_jobs=1, input_type='smiles')
    try:
        fps.transform(data['mols'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'

    try:
        fps.transform(data['smis'])
    except:
        assert False, 'should not got error'
    else:
        assert True


def test_apfp_3(data):
    fps = AtomPairFP(n_jobs=1, input_type='any')
    try:
        fps.transform(data['mols'] + data['smis'])
    except BaseException as e:
        print(e)
        assert False, 'should not got error'
    else:
        assert True


def test_apfp_4(data):
    fps = AtomPairFP(n_jobs=1, input_type='any')
    try:
        fps.transform(data['err_smis'])
    except ValueError:
        assert True
    else:
        assert False, 'should not got error'

    fps = ECFP(n_jobs=1, input_type='any', on_errors='nan')
    try:
        ret = fps.transform(data['err_smis'])
        assert pd.DataFrame(data=ret).shape == (4, 2048)
        assert np.isnan(ret[1][10])
        assert np.isnan(ret[2][20])
    except:
        assert False
    else:
        assert True


def test_rdfp_1(data):
    fps = RDKitFP(n_jobs=1)

    try:
        fps.transform(data['mols'])
    except:
        assert False, 'should not got error'
    else:
        assert True

    try:
        fps.transform(data['smis'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'


def test_rdfp_2(data):
    fps = RDKitFP(n_jobs=1, input_type='smiles')
    try:
        fps.transform(data['mols'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'

    try:
        fps.transform(data['smis'])
    except:
        assert False, 'should not got error'
    else:
        assert True


def test_rdfp_3(data):
    fps = RDKitFP(n_jobs=1, input_type='any')
    try:
        fps.transform(data['mols'] + data['smis'])
    except BaseException as e:
        print(e)
        assert False, 'should not got error'
    else:
        assert True


def test_rdfp_4(data):
    fps = RDKitFP(n_jobs=1, input_type='any')
    try:
        fps.transform(data['err_smis'])
    except ValueError:
        assert True
    else:
        assert False, 'should not got error'

    fps = ECFP(n_jobs=1, input_type='any', on_errors='nan')
    try:
        ret = fps.transform(data['err_smis'])
        assert pd.DataFrame(data=ret).shape == (4, 2048)
        assert np.isnan(ret[1][10])
        assert np.isnan(ret[2][20])
    except:
        assert False
    else:
        assert True


def test_maccs_1(data):
    fps = MACCS(n_jobs=1)

    try:
        fps.transform(data['mols'])
    except:
        assert False, 'should not got error'
    else:
        assert True

    try:
        fps.transform(data['smis'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'


def test_maccs_2(data):
    fps = MACCS(n_jobs=1, input_type='smiles')
    try:
        fps.transform(data['mols'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'

    try:
        fps.transform(data['smis'])
    except:
        assert False, 'should not got error'
    else:
        assert True


def test_maccs_3(data):
    fps = MACCS(n_jobs=1, input_type='any')
    try:
        fps.transform(data['mols'] + data['smis'])
    except BaseException as e:
        print(e)
        assert False, 'should not got error'
    else:
        assert True


def test_maccs4(data):
    fps = MACCS(n_jobs=1, input_type='any')
    try:
        fps.transform(data['err_smis'])
    except ValueError:
        assert True
    else:
        assert False, 'should not got error'

    fps = ECFP(n_jobs=1, input_type='any', on_errors='nan')
    try:
        ret = fps.transform(data['err_smis'])
        assert pd.DataFrame(data=ret).shape == (4, 2048)
        assert np.isnan(ret[1][10])
        assert np.isnan(ret[2][20])
    except:
        assert False
    else:
        assert True


def test_fps_1(data):
    fps = Fingerprints(n_jobs=1)

    try:
        fps.transform(data['mols'])
    except:
        assert False, 'should not got error'
    else:
        assert True

    try:
        fps.transform(data['smis'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'


def test_fps_2(data):
    fps = Fingerprints(n_jobs=1, input_type='smiles')
    try:
        fps.transform(data['mols'])
    except TypeError:
        assert True
    else:
        assert False, 'should not got error'

    try:
        fps.transform(data['smis'])
    except BaseException as e:
        print(e)
        assert False, 'should not got error'
    else:
        assert True


def test_fps_3(data):
    fps = Fingerprints(n_jobs=1, input_type='any')
    try:
        fps.transform(data['mols'] + data['smis'])
    except BaseException as e:
        print(e)
        assert False, 'should not got error'
    else:
        assert True


def test_fps_4(data):
    fps = Fingerprints(n_jobs=1, input_type='any')
    try:
        fps.transform(data['err_smis'])
    except ValueError:
        assert True
    else:
        assert False, 'should not got error'

    fps = ECFP(n_jobs=1, input_type='any', on_errors='nan')
    try:
        ret = fps.transform(data['err_smis'])
        assert pd.DataFrame(data=ret).shape == (4, 2048)
        assert np.isnan(ret[1][10])
        assert np.isnan(ret[2][20])
    except:
        assert False
    else:
        assert True


if __name__ == "__main__":
    pytest.main()
