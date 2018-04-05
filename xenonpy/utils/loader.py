# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from pathlib import Path
from urllib.parse import urljoin, urlparse
from warnings import warn

import pandas as pd
import requests
from ruamel.yaml import YAML

from .config import get_conf, get_dataset_url
from .dataset import DataSet
from .functional import absolute_path, get_sha256
from .. import __cfg_root__


class Loader(object):
    """
    Load data from embed dataset in XenonPy's or user create data saved in ``~/.xenonpy/cached`` dir.
    Also can fetch data by http request.

    This is sample to demonstration how to use is. Also see parameters documents for details.

    ::

        >>> load = Loader()
        >>> elements = load('elements')
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
    dataset = ('elements', 'elements_completed', 'mp_inorganic',
               'mp_structure')
    # set to check params

    _dataset_dir = Path().home() / __cfg_root__ / 'dataset'
    _cached_dir = Path().home() / __cfg_root__ / 'cached'
    _yaml = YAML(typ='safe')
    _yaml.indent(mapping=2, sequence=4, offset=2)

    def __init__(self, location=None):
        """

        Parameters
        ----------
        location: str
            Where data are.
            None(default) means to load data in ``~/.xenonpy/cached`` or ``~/.xenonpy/dataset`` dir.
            When given as url, later can be uesd to fetch files under this url from http-request.
        """
        self._location = location
        self._type = self._select(location)

    def _select(self, loc):
        if loc is None:
            return 'local'

        if not isinstance(loc, str):
            raise TypeError(
                'parameter `location` must be string but got {}'.format(
                    type(loc)))

        # local
        scheme = urlparse(loc).scheme
        if scheme is '':
            self._location = absolute_path(loc)
            return 'user_loc'

        # http
        http_schemes = ('http', 'https')
        if scheme in http_schemes:
            return 'http'

        raise ValueError('can not parse location {}'.format(loc))

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

    def __call__(self, data=None, **kwargs):
        """
        Same to self.load()
        """
        return self.load(data=data, **kwargs)

    def load(self, data=None, **kwargs):
        """
        load data.

        .. note::
            Try to load data from local at ``~/.xenonpy/dataset``.
            If no data, try to fetch them from remote repository.

        Args
        -----------
        data: str
            name of data.

        Returns
        ------
        ret:DataFrame or Saver or local file path.
        """

        kwargs = dict(kwargs, data=data)

        def _get_data(key, ignore_err=False, del_key=True):
            try:
                ret = kwargs[key]
                if del_key:
                    del kwargs[key]
                return ret
            except KeyError:
                if ignore_err:
                    return None
                else:
                    raise KeyError('no key `{}` in **kwargs'.format(key))

        if self._type == 'local':
            data = _get_data('data')
            if data in self.dataset:
                dataset = self._dataset_dir / (data + '.pkl.pd_')
                sha256_file = self._dataset_dir / 'sha256.yml'

                # fetch data from source if not in local
                if not dataset.exists():
                    url = get_dataset_url(data)
                    print('fetching dataset `{}` from {} because of none data file or is older'.format(data, url))
                    print('for some reason you can\'t download it automatically, '
                          'you can download it manually from the url above and '
                          'put it under `~/.xenonpy/dataset/`')
                    self._http_data(url, save_to=str(dataset))

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

                if sha256_ != get_conf(data):
                    warn(
                        'local data {} is different from the repository version.\n'
                        'use `load(data, sync=True)` to fix it.'.format(data),
                        RuntimeWarning)
                    if _get_data('sync', ignore_err=True):
                        url = get_dataset_url(data)
                        print(
                            'fetching dataset `{}` from {} because of none or old'.
                                format(data, url))
                        print('you can download it manually from this url'
                              'then put it under `~/.xenonpy/dataset/`')
                        self._http_data(url, save_to=str(dataset))
                        sha256_ = get_sha256(str(dataset))
                        sha256[data] = sha256_
                        self._yaml.dump(sha256, sha256_file)

                # return preset data
                return pd.read_pickle(str(dataset))

            # return user local data
            return DataSet(data, ignore_err=False)

        if self._type == 'user_loc':
            data = _get_data('data')
            return DataSet(data, path=self._location, ignore_err=False)

        if self._type == 'http':
            data = _get_data('data', ignore_err=True)
            params = _get_data('params', ignore_err=True)
            kwargs['stream'] = True
            url = urljoin(self._location, data)
            return self._http_data(url, params=params, **kwargs)

        raise ValueError('can not fetch data')

    def _get_prop(self, name):
        tmp = self._type
        self._type = 'local'
        ret = self(name)
        self._type = tmp
        return ret

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

        .. _Materials Projects: https://www.materialsproject.org/

        Returns
        -------
        DataFrame:
            Structures as dict that can be loaded by pymatgen.
        """
        # return self._get_prop('mp_structure')
        raise NotImplementedError()

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
