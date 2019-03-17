#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from itertools import zip_longest
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
from pymatgen import MPRester
from ruamel.yaml import YAML
from tqdm import tqdm

from .dataset import Dataset
from .._conf import __cfg_root__
from ..utils import config, get_dataset_url, get_sha256, Singleton


class Preset(Dataset, metaclass=Singleton):
    """
    Load data from embed dataset in XenonPy's or user create data saved in ``~/.xenonpy/cached`` dir.
    Also can fetch data by http request.

    This is sample to demonstration how to use is. Also see parameters documents for details.

    ::

        >>> from xenonpy.datatools import preset
        >>> elements = preset.elements
        >>> elements.info()
        <class 'pandas.core.frame.DataFrame'>
        Index: 118 entries, H to Og
        Data columns (total 74 columns):
        atomic_number                    118 non-null int64
        atomic_radius                    88 non-null float64
        atomic_radius_rahm               96 non-null float64
        atomic_volume                    91 non-null float64
        atomic_weight                    118 non-null float64
        boiling_point                    96 non-null float64
        brinell_hardness                 59 non-null float64
        bulk_modulus                     69 non-null float64
        ...
    """

    __dataset__ = ('elements', 'elements_completed')
    __builder__ = ('mp_samples',)

    # set to check params

    def __init__(self):
        self._dataset = Path(__cfg_root__) / 'dataset'
        self._ext_data = config('ext_data')
        super().__init__(
            str(self._dataset),
            config('userdata'),
            *self._ext_data,
            backend='dataframe',
            prefix=('dataset',))

        yaml = YAML(typ='safe')
        yaml.indent(mapping=2, sequence=4, offset=2)

        self._yaml = yaml

    def sync(self, data, to=None):
        """
        load data.

        .. note::
            Try to load data from local at ``~/.xenonpy/dataset``.
            If no data, try to fetch them from remote repository.

        Args
        -----------
        data: str
            name of data.
        to: str
            The version of repository.
            See Also: https://github.com/yoshida-lab/dataset/releases

        Returns
        ------
        ret:DataFrame or Saver or local file path.
        """

        dataset = self._dataset / (data + '.pkl.pd_')
        sha256_file = self._dataset / 'sha256.yml'

        # check sha256 value
        # make sure sha256_file file exist.
        sha256_file.touch()
        sha256 = self._yaml.load(sha256_file)
        if sha256 is None:
            sha256 = {}

        # fetch data from source if not in local
        if not to:
            url = get_dataset_url(data)
        else:
            url = get_dataset_url(data, to)
        print('fetching dataset `{0}` from {1}.'.format(data, url))
        self.from_http(url, save_to=str(self._dataset))

        sha256_ = get_sha256(str(dataset))
        sha256[data] = sha256_
        self._yaml.dump(sha256, sha256_file)

        self._make_index(prefix=['dataset'])

    def build(self, *keys, save_to=None, **kwargs):

        # build materials project dataset
        def mp_builder(api_key, mp_ids):

            #     print('Will fetch %s inorganic compounds from Materials Project' % len(mp_ids))

            # split requests into fixed number groups
            # eg: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
            def grouper(iterable, n, fillvalue=None):
                """Collect data into fixed-length chunks or blocks"""
                args = [iter(iterable)] * max(n, 1)
                return zip_longest(fillvalue=fillvalue, *args)

            # the following props will be fetched
            mp_props = [
                'band_gap',
                'density',
                'volume',
                'material_id',
                'pretty_formula',
                'elements',
                'efermi',
                'e_above_hull',
                'formation_energy_per_atom',
                'final_energy_per_atom',
                'unit_cell_formula',
                'structure'
            ]

            entries = []
            mpid_groups = [g for g in grouper(mp_ids, len(mp_ids) // 10)]

            with MPRester(api_key) as mpr:
                for group in tqdm(mpid_groups):
                    mpid_list = [id for id in filter(None, group)]
                    chunk = mpr.query({"material_id": {"$in": mpid_list}}, mp_props)
                    entries.extend(chunk)

            df = pd.DataFrame(entries, index=[e['material_id'] for e in entries])
            df = df.drop('material_id', axis=1)
            df = df.rename(columns={'unit_cell_formula': 'composition'})
            df = df.reindex(columns=sorted(df.columns))

            return df

        for key in keys:
            if key is 'mp_samples':
                if 'api_key' not in kwargs:
                    raise RuntimeError('api key of materials projects database is needed')
                if 'mp_ids' in kwargs:
                    ids = kwargs['mp_ids']
                    if isinstance(ids, (list, tuple)):
                        mp_ids = ids
                    elif isinstance(ids, str):
                        mp_ids = [s.decode('utf-8') for s in np.loadtxt(ids, 'S20')]
                    else:
                        raise ValueError(
                            'parameter `mp_ids` can only be a str to specific the ids file path'
                            'or a list-like object contain the ids')
                else:
                    ids = Path(__file__).absolute().parents[2] / 'samples' / 'mp_ids.txt'
                    mp_ids = [s.decode('utf-8') for s in np.loadtxt(str(ids), 'S20')]
                data = mp_builder(kwargs['api_key'], mp_ids)
                if not save_to:
                    save_to = Path(config('userdata')) / 'mp_samples.pkl.pd_'
                    save_to = save_to.expanduser().absolute()
                data.to_pickle(save_to)
                self._make_index(prefix=['dataset'])
                return

        raise ValueError('no available key(s) in %s, these can only be %s' % (keys, self.__builder__))

    def _check(self, data):

        dataset = self._dataset / (data + '.pkl.pd_')
        sha256_file = self._dataset / 'sha256.yml'

        # fetch data from source if not in local
        if not dataset.exists():
            raise RuntimeError(
                "data {0} not exist, please run <Preset.sync('{0}')> to download from the repository".format(data))

        # check sha256 value
        sha256_file.touch()  # make sure sha256_file file exist.
        sha256 = self._yaml.load(sha256_file)
        if sha256 is None:
            sha256 = {}
        if data not in sha256:
            sha256_ = get_sha256(str(dataset))
            sha256[data] = sha256_
            self._yaml.dump(sha256, sha256_file)
        else:
            sha256_ = sha256[data]

        if sha256_ != config(data):
            warn(
                "local version {0} is different from the latest version {1}."
                "you can use <Preset.sync('{0}', to='{1}')> to fix it.".format(data, config('db_version')),
                RuntimeWarning)

    @property
    def elements(self):
        """
        Element properties from embed dataset.
        These properties are summarized from `mendeleev`_, `pymatgen`_, `CRC Handbook`_ and `magpie`_.

        See Also: :doc:`features`

        .. _mendeleev: https://mendeleev.readthedocs.io
        .. _pymatgen: http://pymatgen.org/
        .. _CRC Handbook: http://hbcponline.com/faces/contents/ContentsSearch.xhtml
        .. _magpie: https://bitbucket.org/wolverton/magpie

        Returns
        -------
        DataFrame:
            element properties in pd.DataFrame
        """
        self._check('elements')
        return self.dataset_elements

    @property
    def elements_completed(self):
        """
        Completed element properties. [MICE]_ imputation used

        .. [MICE] `Int J Methods Psychiatr Res. 2011 Mar 1; 20(1): 40–49.`__
                    doi: `10.1002/mpr.329 <10.1002/mpr.329>`_

        .. __: https://www.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&retmode=ref&cmd=prlinks&id=21499542

        See Also: :doc:`features`

        Returns
        -------
            imputed element properties in pd.DataFrame
        """
        self._check('elements_completed')
        return self.dataset_elements_completed


preset = Preset()
