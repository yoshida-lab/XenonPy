#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from pathlib import Path

import pandas as pd
import pytest
import torch
from pymatgen import Structure as pmg_S

from xenonpy.descriptor import CrystalGraphFeaturizer
from xenonpy.descriptor.base import BaseGraphFeaturizer


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


def test_base_1(data):
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseGraphFeaturizer()


def test_crystal_1(data):
    cg = CrystalGraphFeaturizer()
    assert cg.feature_labels == ['atom_feature', 'bond_feature']

    tmp = cg.node_features(data[0])
    assert isinstance(tmp, torch.Tensor)
    assert tmp.shape == (16, 92)

    edges, ids = cg.edge_features(data[0])
    assert isinstance(edges, torch.Tensor)
    assert edges.shape == (16, 12, 41)

    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (16, 12)


def test_crystal_2(data):
    cg = CrystalGraphFeaturizer(max_num_nbr=15, radius=10, atom_feature='elements')

    tmp = cg.node_features(data[0])
    assert isinstance(tmp, torch.Tensor)
    assert tmp.shape == (16, 58)

    edges, ids = cg.edge_features(data[0])
    assert isinstance(edges, torch.Tensor)
    assert edges.shape == (16, 15, 51)

    assert isinstance(ids, torch.Tensor)
    assert ids.shape == (16, 15)


def test_crystal_3(data):
    cg = CrystalGraphFeaturizer()
    tmp = cg.transform(data)

    assert isinstance(tmp, pd.DataFrame)
    assert tmp.shape == (2, 2)
    assert tmp.values[0, 0].shape == (16, 92)
    assert tmp.values[0, 1][0].shape == (16, 12, 41)
    assert tmp.values[0, 1][1].shape == (16, 12)


def test_crystal_4(data):
    cg = CrystalGraphFeaturizer(atom_feature=lambda s: [1, 2, 3, 4], n_jobs=1)
    tmp = cg.transform(data)

    assert isinstance(tmp, pd.DataFrame)
    assert tmp.shape == (2, 2)
    assert tmp.values[0, 0][0].numpy().tolist() == [1, 2, 3, 4]


if __name__ == "__main__":
    pytest.main()
