#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


from os import remove
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.externals import joblib

from xenonpy.datatools import Dataset


@pytest.fixture(scope='module')
def test_data():
    # ignore numpy warning
    import warnings
    print('ignore NumPy RuntimeWarning\n')
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

    file_path = Path(__file__).parent
    file_name = 'rename.txt'
    file_url = 'https://raw.githubusercontent.com/yoshida-lab/XenonPy/master/travis/fetch_test.txt'

    # create data
    ary = [[1, 2], [3, 4]]
    df = pd.DataFrame(ary)

    pkl_path = str(file_path / 'test.pkl.z_')
    df_path = str(file_path / 'test.pkl.pd_')
    csv_path = str(file_path / 'test.csv')

    joblib.dump(ary, pkl_path)
    df.to_csv(csv_path)
    df.to_pickle(df_path)

    yield (file_name, file_url, file_path)

    tmp = file_path / file_name
    if tmp.exists():
        remove(str(tmp))

    tmp = file_path / 'fetch_test.txt'
    if tmp.exists():
        remove(str(tmp))

    remove(pkl_path)
    remove(df_path)
    remove(csv_path)

    print('test over')


def test_dataset_1(test_data):
    path = Path(__file__).parents[0]
    ds = Dataset(str(path), backend='pickle')
    assert hasattr(ds, 'datatools_test')

    tmp = ds.datatools_test
    assert isinstance(tmp, list)
    assert tmp == [[1, 2], [3, 4]]

    tmp = ds.csv
    assert hasattr(tmp, 'datatools_test')
    tmp = tmp.datatools_test
    assert isinstance(tmp, pd.DataFrame)
    assert np.all(np.array([[0, 1, 2], [1, 3, 4]]) == tmp.values)

    tmp = ds.csv(str(path / 'test.csv'))
    assert np.all(np.array([[0, 1, 2], [1, 3, 4]]) == tmp.values)

    tmp = ds.dataframe
    assert hasattr(tmp, 'datatools_test')
    tmp = tmp.datatools_test
    assert isinstance(tmp, pd.DataFrame)
    assert np.all(np.array([[1, 2], [3, 4]]) == tmp.values)

    tmp = ds.dataframe(str(path / 'test.pkl.pd_'))
    assert np.all(np.array([[1, 2], [3, 4]]) == tmp.values)


def test_dataset_2(test_data):
    with pytest.raises(RuntimeError, match='is not a legal path'):
        Dataset.from_http(test_data[1], 'not_exist')

    tmp = Dataset.from_http(test_data[1], save_to=test_data[2])
    assert tmp == str(test_data[2] / 'fetch_test.txt')
    assert Path(tmp).exists()
    with open(tmp, 'r') as f:
        assert f.readline() == 'Test xenonpy.utils.Loader._fetch_data'

    tmp = Dataset.from_http(test_data[1], save_to=test_data[2], filename=test_data[0])
    assert tmp == str(test_data[2] / 'rename.txt')
    assert Path(tmp).exists()
    with open(tmp, 'r') as f:
        assert f.readline() == 'Test xenonpy.utils.Loader._fetch_data'


if __name__ == "__main__":
    pytest.main()
