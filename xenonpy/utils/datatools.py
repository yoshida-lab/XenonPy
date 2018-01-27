# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import uuid
from collections import defaultdict
from datetime import datetime as dt
from os import remove
from os.path import getmtime
from pathlib import Path
from shutil import rmtree
from urllib.parse import urlparse

import pandas as pd
import requests
from sklearn.externals import joblib as jl

from .. import cfg_root
from .. import _get_dataset_url


class Loader(object):
    """
    Load data from embed dataset in XenonPy's or user create data saved in ``~/.xenonpy/cached`` dir.
    Also can fetch data by http request.

    This is sample to demonstration how to use is. Also see parameters documents for details.

    ::

        >>> load = Loader()
        >>> elements = load('elements')
        >>> ele.info()
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
        c6                               43 non-null float64
        c6_gb                            86 non-null float64
        covalent_radius_bragg            37 non-null float64
        covalent_radius_cordero          96 non-null float64
        covalent_radius_pyykko           118 non-null int64
        covalent_radius_pyykko_double    108 non-null float64
        covalent_radius_pyykko_triple    80 non-null float64
        covalent_radius_slater           86 non-null float64
        density                          95 non-null float64
        dipole_polarizability            106 non-null float64
        electron_negativity              103 non-null float64
        electron_affinity                77 non-null float64
        en_allen                         71 non-null float64
        en_ghosh                         103 non-null float64
        en_pauling                       85 non-null float64
        first_ion_en                     103 non-null float64
        fusion_enthalpy                  92 non-null float64
        gs_bandgap                       112 non-null float64
        gs_energy                        112 non-null float64
        gs_est_bcc_latcnt                112 non-null float64
        gs_est_fcc_latcnt                112 non-null float64
        gs_mag_moment                    112 non-null float64
        gs_volume_per                    112 non-null float64
        hhi_p                            77 non-null float64
        hhi_r                            77 non-null float64
        heat_capacity_mass               85 non-null float64
        heat_capacity_molar              85 non-null float64
        icsd_volume                      112 non-null float64
        evaporation_heat                 88 non-null float64
        gas_basicity                     32 non-null float64
        heat_of_formation                89 non-null float64
        lattice_constant                 87 non-null float64
        linear_expansion_coefficient     63 non-null float64
        mendeleev_number                 103 non-null float64
        melting_point                    100 non-null float64
        metallic_radius                  56 non-null float64
        metallic_radius_c12              63 non-null float64
        molar_volume                     97 non-null float64
        num_unfilled                     112 non-null float64
        num_valance                      112 non-null float64
        num_d_unfilled                   112 non-null float64
        num_d_valence                    112 non-null float64
        num_f_unfilled                   112 non-null float64
        num_f_valence                    112 non-null float64
        num_p_unfilled                   112 non-null float64
        num_p_valence                    112 non-null float64
        num_s_unfilled                   112 non-null float64
        num_s_valence                    112 non-null float64
        period                           118 non-null int64
        poissons_ratio                   54 non-null float64
        proton_affinity                  32 non-null float64
        specific_heat                    81 non-null float64
        thermal_conductivity             96 non-null float64
        vdw_radius                       103 non-null float64
        vdw_radius_alvarez               94 non-null float64
        vdw_radius_batsanov              65 non-null float64
        vdw_radius_bondi                 28 non-null float64
        vdw_radius_dreiding              21 non-null float64
        vdw_radius_mm3                   94 non-null float64
        vdw_radius_rt                    9 non-null float64
        vdw_radius_truhlar               16 non-null float64
        vdw_radius_uff                   103 non-null float64
        sound_velocity                   72 non-null float64
        vickers_hardness                 39 non-null float64
        Polarizability                   102 non-null float64
        youngs_modulus                   63 non-null float64
        dtypes: float64(71), int64(3)
        memory usage: 69.1+ KB
    """

    # set to check params

    def __init__(self,
                 *,
                 chunk_size: int = 256 * 1024,
                 api_key: str = None,
                 username: str = None,
                 password: str = None,
                 payload: dict = None):
        """

        Parameters
        ----------
        url: str
            Where data are.
            None(default) means to load data in ``~/.xenonpy/cached`` or ``~/.xenonpy/dataset`` dir.
            When given as url, later can be uesd to fetch files under this url from http-request.

        chunk_size: int
            Http-request chunk size.

        api_key: str
            If remote access need a api key.

        username: str
            If need user name.

        password: str
            If need password.

        payload: dict
            If need payload.

        """
        self.payload = payload
        self.pass_word = password
        self.user_name = username
        self.api_key = api_key
        self.datasets = [
            'elements', 'elements_completed', 'mp_inorganic',
            'electron_density', 'sample_A', 'mp_structure'
        ]
        self.chunk_size = chunk_size
        self.dataset_dir = Path().home() / cfg_root / 'dataset'
        self.user_data_dir = Path().home() / cfg_root / 'userdata'
        self.cached_dir = Path().home() / cfg_root / 'cached'

    def _fetch_data(self, url, save_to=None):
        schemes = {'http', 'https'}
        scheme = urlparse(url).scheme
        if 'http' in scheme:
            return self._http_data(url, save_to)
        else:
            raise ValueError("Only can access [{}] data but you send {}. :(".format(schemes, scheme))

    def _http_data(self, url, save_to=None):
        r = requests.get(url, stream=True)
        r.raise_for_status()

        if 'filename' in r.headers:
            filename = r.headers['filename']
        else:
            filename = url.split('/')[-1]

        save_to = str(self.cached_dir / filename) if not save_to else str(save_to)
        with open(save_to, 'wb') as f:
            for chunk in r.iter_content(chunk_size=self.chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        return save_to

    def __call__(self, dname: str):
        """
        load preset dataset.

        .. note::
            Try to load data from local at ``~/.xenonpy/dataset``.
            If no data, try to fetch them from remote repository.

        Args
        -----------
        dname: str
            name of dateset.

        Returns
        ------
        ret:DataFrame or Saver or local file path.
        """

        # check dataset name
        if dname in self.datasets:
            dataset = self.dataset_dir / (dname + '.pkl.pd_')

            # check dataset exist
            if not dataset.exists():
                self._fetch_data(_get_dataset_url(dname), dataset)

            # fetch data from source
            return pd.read_pickle(str(dataset))

        elif '/' not in dname:
            dataset = self.user_data_dir / dname
            if not dataset.is_dir():
                raise FileNotFoundError(
                    'no user dataset under {}'.format(dataset))
            return Saver(dname)

        else:
            file_name = self._fetch_data(dname)
            dataset = self.cached_dir / file_name
            return dataset

    @property
    def elements(self):
        """
        Element properties from embed dataset.
        These properties are summarized from `mendeleev`_, `pymatgen`_, `CRC Handbook`_ and `magpie`_.

        See Also: :doc:`dataset`

        .. _mendeleev: https://mendeleev.readthedocs.io
        .. _pymatgen: http://pymatgen.org/
        .. _CRC Handbook: http://hbcponline.com/faces/contents/ContentsSearch.xhtml
        .. _magpie: https://bitbucket.org/wolverton/magpie

        Returns
        -------
        DataFrame:
            element properties in pd.DataFrame
        """
        return self('elements')

    @property
    def elements_completed(self):
        """
        Completed element properties. [MICE]_ imputation used

        .. [MICE] `Int J Methods Psychiatr Res. 2011 Mar 1; 20(1): 40–49.`__
                    doi: `10.1002/mpr.329 <10.1002/mpr.329>`_

        .. __: https://www.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi?dbfrom=pubmed&retmode=ref&cmd=prlinks&id=21499542

        See Also: :doc:`dataset`

        Returns
        -------
            imputed element properties in pd.DataFrame
        """
        return self('elements_completed')


class Saver(object):
    """
    Save data in a convenient way:

    .. code:: python

        import numpy as np
        np.random.seed(0)

        # some data
        data1 = np.random.randn(5, 5)
        data2 = np.random.randint(5, 5)

        # init Saver
        save = Saver('you_dataset_name')

        # save data
        save(data1, data2)

        # retriever data
        date = save.last()  # last saved
        data = save[0]  # by index
        for data in save:  # as iterator
            do_something(data)

        # delete data
        save.delete(0)  # by index
        save.delete()  # delete 'you_dataset_name' dir

    See Also: :doc:`dataset`
    """

    def __init__(self, dataset=None):
        """
        Parameters
        ----------
        dataset: str
            The dir to save and load data
        """
        self.dataset = dataset
        self._path = Path.home() / cfg_root / 'userdata' / dataset
        if not self._path.exists():
            self._path.mkdir(parents=True)
        self._files = None
        self._make_file_index()

    def _make_file_index(self):
        self._files = defaultdict(list)
        files = [f for f in self._path.iterdir() if f.match('*.pkl.*')]

        for f in files:
            # named data
            if len(f.suffixes) == 3:
                d_name = f.suffixes[0].lstrip('.')
                self._files[d_name].append(f)
                continue

            # unnamed data
            self._files['temp_data'].append(f)

        for v in self._files.values():
            self._sort_files(v)

    @classmethod
    def _sort_files(cls, files):
        if files is not None:
            files.sort(key=lambda f: getmtime(str(f)))

    @classmethod
    def _load_data(cls, file):
        if file.suffix == '.pd_':
            return pd.read_pickle(str(file))
        else:
            return jl.load(file)

    def _save_data(self, data, filename=None):
        uid = str(uuid.uuid1()).replace('-', '')
        prefix = uid + '.' + filename if filename else uid
        if not self._path.exists():
            self._path.mkdir()

        if isinstance(data, pd.DataFrame):
            file = self._path / (prefix + '.pkl.pd_')
            pd.to_pickle(data, str(file))
        else:
            file = self._path / (prefix + '.pkl.z')
            jl.dump(data, file)

        return file

    def dump(self, fpath: str, *,
             fname: str = None,
             with_datetime: bool = True):
        """
        Dump last checked dataset to file.

        Parameters
        ----------
        fpath: str
            Where save to.
        fname: str
            File name. Omit to use dataset name.
        with_datetime: bool
            Suffix file name with dumped time.

        Returns
        -------
        ret: str
            File path.
        """
        ret = {k: self._load_data(v[-1]) for k, v in self._files.items()}
        name = fname if fname else self.dataset
        if with_datetime:
            datetime = dt.now().strftime('-%Y-%m-%d_%H-%M-%S_%f')
        else:
            datetime = ''
        path_dir = Path(fpath).expanduser()
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
        path = path_dir / (name + datetime + '.pkl.z')
        jl.dump(ret, path)

        return str(path)

    def last(self, d_name: str = None):
        """
        Return last saved data.

        Args
        ----
        d_name: str
            Data's name. Omit for access temp data

        Return
        -------
        ret:any python object
            Data stored in `*.pkl` file.
        """
        if d_name is None:
            return self._load_data(self._files['temp_data'][-1])
        return self._load_data(self._files[d_name][-1])

    def rm(self, index, d_name: str = None):
        """
        Delete file(s) with given index.

        Parameters
        ----------
        index: int or slice
            Index of data. Data sorted by datetime.
        d_name: str
            Data's name. Omit for access unnamed data.
        """

        if not d_name:
            files = self._files['temp_data'][index]
            if not isinstance(files, list):
                remove(str(files))
            else:
                for f in files:
                    remove(str(f))
            del self._files['temp_data'][index]
            return

        files = self._files[d_name][index]
        if not isinstance(files, list):
            remove(str(files))
        else:
            for f in files:
                remove(str(f))
        del self._files[d_name][index]

    def clean(self, data_name: str = None):
        """
        Remove all data by name. Omit to remove hole dataset.

        Parameters
        ----------
        data_name: str
            Data's name.Omit to remove hole dataset.
        """
        if data_name is None:
            rmtree(str(self._path))
            self._files = list()
            self._files = defaultdict(list)
            return

        for f in self._files[data_name]:
            remove(str(f))
        del self._files[data_name]

    def __repr__(self):
        cont_ls = []
        cont_ls.append('"{}" include:'.format(self.dataset))

        for k, v in self._files.items():
            cont_ls.append('"{}": {}'.format(k, len(v)))

        return '\n'.join(cont_ls)

    def __getitem__(self, item):

        # load file
        def _load_file(files, item):
            _files = files[item]
            if not isinstance(_files, list):
                return self._load_data(_files)
            return [self._load_data(f) for f in _files]

        if isinstance(item, tuple):
            try:
                key, index = item
            except ValueError:
                raise ValueError('except 2 parameters. [str, int or slice]')
            if not isinstance(key, str) or \
                    (not isinstance(index, int) and not isinstance(index, slice)):
                raise ValueError('except 2 parameters. [str, int or slice]')
            return _load_file(self._files[key], index)

        if isinstance(item, str):
            return self.__getitem__((item, slice(None, None, None)))

        return _load_file(self._files['temp_data'], item)

    def __call__(self, *data, **named_data):
        for d in data:
            f = self._save_data(d)
            self._files['temp_data'].append(f)

        for k, v in named_data.items():
            f = self._save_data(v, k)
            self._files[k].append(f)
