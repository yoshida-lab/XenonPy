# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import pandas as pd

from xenonpy import Path, PACKAGE_CONF_DIR, _get_binary_file_url


def load(name: str, include=None, exclude=None):
    """
    load preset dataset.

    .. note::
        loader will try to load data locally if there they are.
        If no data, try to fetch them from repository

    Args
    -----------
    name: str
        name of dateset.
    include: str list
        filter which columns should be included.
    exclude: str list
        filter which columns should drop out.

    Returns
    ------
    pandas.DataFrame
        return loaded data in panda.DataFrame object.
    """

    # set to check params
    _datasets = ['elements', 'mp_inorganic', 'electron_density', 'sample_A', 'mp_structure']

    # check dataset name
    if not isinstance(name, str) or name not in _datasets:
        raise ValueError("param 'dataset' must be a known name.")

    dataset_dir = Path().home() / PACKAGE_CONF_DIR / 'dataset'
    dataset = dataset_dir / (name + '.pkl')

    # check dataset exist
    if not dataset.exists():
        import urllib3
        from urllib3.exceptions import HTTPError
        bin_url = _get_binary_file_url(name)
        http = urllib3.PoolManager()
        chunk_size = 256 * 1024

        # fetch `name.pkl` file
        try:
            r = http.request('GET', bin_url, preload_content=False)
            with open(dataset, 'wb') as out:
                while True:
                    data = r.read(chunk_size)
                    if not data:
                        break
                    out.write(data)

        except HTTPError as e:
            print('can not fetch data from {}.'.format(bin_url))
            print(e)
        finally:
            r.release_conn()

    # fetch data from source
    data = pd.read_pickle(str(dataset))
    if include:
        data = data[include]
    if exclude:
        data = data.drop(exclude, axis=1)
    return data


def load_elements():
    return load('elements')


def load_inorganic():
    return load('mp_inorganic')


def load_electron_density():
    return load('electron_density')


def load_structure():
    return load('mp_structure')
