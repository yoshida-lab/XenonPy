#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import re
from collections import defaultdict
from datetime import datetime as dt
from os import remove
from os.path import getmtime
from pathlib import Path
from shutil import rmtree
from warnings import warn

import pandas as pd
import requests
from ruamel.yaml import YAML
from sklearn.externals import joblib

from .._conf import __cfg_root__
from ..utils import Singleton
from ..utils import get_conf, get_dataset_url, get_data_loc, absolute_path, get_sha256


class LocalStorage(object):
    """
    Save data in a convenient way:

    .. code:: python

        import numpy as np
        np.random.seed(0)

        # some data
        data1 = np.random.randn(5, 5)
        data2 = np.random.randint(5, 5)

        # init Saver
        saver = LocalStorage('you_dataset_name')

        # save data
        saver(data1, data2)
        # or
        saver.save(data1, data2)

        # retriever data
        date = saver.last()  # last saved
        data = saver[0]  # by index
        for data in saver:  # as iterator
            do_something(data)

        # delete data
        save.rm(0)  # by index
        save.rm()  # delete 'you_dataset_name' dir

    See Also: :doc:`dataset`
    """

    def __init__(self, name=None, *, path=None, mk_dir=True, backend=None):
        """
        Parameters
        ----------
        name: str
            Name of dataset. Usually this is dir name contains data.
        path: str
            Absolute dir path.
        mk_dir: bool
            Ignore ``FileNotFoundError``.
        """
        self._backend = backend or joblib
        self._name = name
        if path is not None:
            self._path = Path(absolute_path(path, mk_dir)) / name
        else:
            self._path = Path(get_data_loc('userdata')) / name
        if not self._path.exists():
            if not mk_dir:
                raise FileNotFoundError()
            self._path.mkdir(parents=True)
        self._files = None
        self._make_file_index()

    @property
    def name(self):
        """
        Dataset name.

        Returns
        -------
        str
        """
        return self._name

    @property
    def path(self):
        """
        Dataset path

        Returns
        -------
        str
        """
        return str(self._path.parent)

    def _make_file_index(self):
        self._files = defaultdict(list)
        files = [f for f in self._path.iterdir() if f.match('*.pkl.*')]

        for f in files:
            # select data
            fn = '.'.join(f.name.split('.')[:-3])

            # for compatibility
            # fixme: will be removed at future
            if fn == 'unnamed':
                warn('file like `unnamed.@x` will be renamed to `@x`.', RuntimeWarning)
                new_name = '.'.join(f.name.split('.')[-3:])
                new_path = f.parent / new_name
                f.rename(new_path)
                f = new_path

            if fn == '':
                fn = 'unnamed'
            self._files[fn].append(f)

        for fs in self._files.values():
            if fs is not None:
                fs.sort(key=lambda f: getmtime(str(f)))

    def _load_data(self, file):
        if file.suffix == '.pd_':
            return pd.read_pickle(str(file))
        else:
            return self._backend.load(str(file))

    def _save_data(self, data, filename):
        self._path.mkdir(parents=True, exist_ok=True)
        if isinstance(data, pd.DataFrame):
            file = self._path / (filename + '.pkl.pd_')
            pd.to_pickle(data, str(file))
        else:
            file = self._path / (filename + '.pkl.z')
            self._backend.dump(data, str(file))

        return file

    def dump(self, fpath, *, rename=None, with_datetime=True):
        """
        Dump last checked dataset to file.

        Parameters
        ----------
        fpath: str
            Where save to.
        rename: str
            Rename pickle file. Omit to use dataset as name.
        with_datetime: bool
            Suffix file name with dumped time.

        Returns
        -------
        ret: str
            File path.
        """
        ret = {k: self._load_data(v[-1]) for k, v in self._files.items()}
        name = rename if rename else self._name
        if with_datetime:
            datetime = dt.now().strftime('-%Y-%m-%d_%H-%M-%S_%f')
        else:
            datetime = ''
        path_dir = Path(fpath).expanduser()
        if not path_dir.exists():
            path_dir.mkdir(parents=True, exist_ok=True)
        path = path_dir / (name + datetime + '.pkl.z')
        self._backend.dump(ret, str(path))

        return str(path)

    def last(self, name=None):
        """
        Return last saved data.

        Args
        ----
        name: str
            Data's name. Omit for access temp data

        Return
        -------
        ret:any python object
            Data stored in `*.pkl` file.
        """
        try:
            if name is None:
                return self._load_data(self._files['unnamed'][-1])
            if not isinstance(name, str):
                raise TypeError('para `name` must be string but got %s' % type(name))
            return self._load_data(self._files[name][-1])
        except IndexError:
            return None

    def rm(self, index, name=None):
        """
        Delete file(s) with given index.

        Parameters
        ----------
        index: int or slice
            Index of data. Data sorted by datetime.
        name: str
            Data's name. Omit for access unnamed data.
        """

        if not name:
            files = self._files['unnamed'][index]
            if not isinstance(files, list):
                remove(str(files))
            else:
                for f in files:
                    remove(str(f))
            del self._files['unnamed'][index]
            return

        files = self._files[name][index]
        if not isinstance(files, list):
            remove(str(files))
        else:
            for f in files:
                remove(str(f))
        del self._files[name][index]

    def clean(self, name=None):
        """
        Remove all data by name. Omit to remove hole dataset.

        Parameters
        ----------
        name: str
            Data's name.Omit to remove hole dataset.
        """
        if name is None:
            rmtree(str(self._path))
            self._files = list()
            self._files = defaultdict(list)
            return

        for f in self._files[name]:
            remove(str(f))
        del self._files[name]

    def __getattr__(self, name):
        """
        Returns sub-dataset.

        Parameters
        ----------
        name: str
            Dataset name.

        Returns
        -------
        self
        """
        sub_set = LocalStorage(name, path=str(self._path), backend=self._backend)
        setattr(self, name, sub_set)
        return sub_set

    def __repr__(self):
        cont_ls = ['<{}> include:'.format(self._name)]

        for k, v in self._files.items():
            cont_ls.append('"{}": {}'.format(k, len(v)))

        return '\n'.join(cont_ls)

    def __getitem__(self, item):

        # load file
        def _load_file(files, item_):
            _files = files[item_]
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

        return _load_file(self._files['unnamed'], item)

    def __call__(self, *unnamed_data, **named_data):
        """
        Same to self.save()
        """
        self.save(*unnamed_data, **named_data)

    def save(self, *unnamed_data, **named_data):
        """
        Save data with or without name.
        Data with same name will not be overwritten.

        Parameters
        ----------
        unnamed_data: any object
            Unnamed data.
        named_data: dict
            Named data as k,v pair.

        """

        def _get_file_index(fn):
            if len(self._files[fn]) != 0:
                return int(re.findall(r'@\d+\.', str(self._files[fn][-1]))[-1][1:-1])
            return 0

        num = 0
        for d in unnamed_data:
            if num == 0:
                num = _get_file_index('unnamed')
            num += 1
            f = self._save_data(d, '@' + str(num))
            self._files['unnamed'].append(f)

        for k, v in named_data.items():
            num = _get_file_index(k) + 1
            f = self._save_data(v, k + '.@' + str(num))
            self._files[k].append(f)


class Preset(Singleton):
    """
    Load data from embed dataset in XenonPy's or user create data saved in ``~/.xenonpy/cached`` dir.
    Also can fetch data by http request.

    This is sample to demonstration how to use is. Also see parameters documents for details.

    ::

        >>> from xenonpy.datatools import preset
        >>> elements = preset.load('elements')
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

    # dataset = (
    #     'elements', 'elements_completed', 'mp_inorganic',
    #     'electron_density', 'sample_A', 'mp_structure'
    # )
    dataset = ('elements', 'elements_completed', 'mp_inorganic', 'mp_structure')
    # set to check params

    _dataset_dir = Path().home() / __cfg_root__ / 'dataset'
    _cached_dir = Path().home() / __cfg_root__ / 'cached'
    _yaml = YAML(typ='safe')
    _yaml.indent(mapping=2, sequence=4, offset=2)

    @classmethod
    def _http_data(cls, url, params=None, save_to=None, **kwargs):
        r = requests.get(url, params, **kwargs)
        r.raise_for_status()

        if 'filename' in r.headers:
            filename = r.headers['filename']
        else:
            filename = url.split('/')[-1]

        if save_to is None:
            save_to = str(cls._cached_dir / filename)
        with open(save_to, 'wb') as f:
            for chunk in r.iter_content(chunk_size=256 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        return save_to

    def __call__(self, data, *, sync=False):
        """
        Same to self.load()
        """
        return self.load(data=data, sync=sync)

    @classmethod
    def load(cls, data, *, sync=False):
        """
        load data.

        .. note::
            Try to load data from local at ``~/.xenonpy/dataset``.
            If no data, try to fetch them from remote repository.

        Args
        -----------
        data: str
            name of data.
        sync: bool
            If ``true``, try to sync data file with online version.

        Returns
        ------
        ret:DataFrame or Saver or local file path.
        """

        dataset = cls._dataset_dir / (data + '.pkl.pd_')
        sha256_file = cls._dataset_dir / 'sha256.yml'

        # fetch data from source if not in local
        if not dataset.exists():
            url = get_dataset_url(data)
            print('fetching dataset `{}` from {} because of none data file or outdated. '
                  'If you can\'t download it automatically, '
                  'just download it manually from the url above and '
                  'put it under `~/.xenonpy/dataset/`'.format(data, url))
            cls._http_data(url, save_to=str(dataset))

        # check sha256 value
        sha256_file.touch()  # make sure sha256_file file exist.
        sha256 = cls._yaml.load(sha256_file)
        if sha256 is None:
            sha256 = {}
        if data not in sha256:
            sha256_ = get_sha256(str(dataset))
            sha256[data] = sha256_
            cls._yaml.dump(sha256, sha256_file)
        else:
            sha256_ = sha256[data]

        if sha256_ != get_conf(data):
            warn(
                'local data {} is different from the repository version. '
                'use `load(data, sync=True)` to fix it.'.format(data), RuntimeWarning)
            if sync:
                url = get_dataset_url(data)
                print('fetching dataset `{}` from {} because of none data file or outdated. '
                      'you can download it manually from this url '
                      'then put it under `~/.xenonpy/dataset/`'.format(data, url))
                cls._http_data(url, save_to=str(dataset))
                sha256_ = get_sha256(str(dataset))
                sha256[data] = sha256_
                cls._yaml.dump(sha256, sha256_file)

        # return preset data
        return pd.read_pickle(str(dataset))

    @classmethod
    def _get_prop(cls, name):
        return cls.load(name)

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
        return self._get_prop('elements')

    @property
    def mp_inorganic(self):
        """
        Inorganic properties summarized from `Materials Projects`_.

        .. _Materials Projects: https://www.materialsproject.org/

        Returns
        -------
        DataFrame:
            Properties in pd.DataFrame
        """
        return self._get_prop('mp_inorganic')

    @property
    def mp_structure(self):
        """
        Inorganic structures summarized from `Materials Projects`_.

        Returns
        -------
        DataFrame:
            Structures as dict that can be loaded by pymatgen.
        """
        return self._get_prop('mp_structure')

    @property
    def oqmd_inorganic(self):
        """
        Inorganic properties summarized from `OQMD`_.

        .. _OQMD: http://www.oqmd.org

        Returns
        -------
        DataFrame:
            Properties in pd.DataFrame
        """
        return self._get_prop('oqmd_inorganic')

    @property
    def oqmd_structure(self):
        """
        Inorganic structures summarized from `OQMD`_.

        Returns
        -------
        DataFrame:
            Structures as dict that can be loaded by pymatgen.
        """
        return self._get_prop('oqmd_structure')

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
        return self._get_prop('elements_completed')


preset = Preset()
