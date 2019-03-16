#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from pathlib import Path
from warnings import warn

from ruamel.yaml import YAML

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

    # set to check params

    def __init__(self):
        self._dataset = Path(__cfg_root__) / 'dataset'
        self._userdata = config('userdata')
        self._ext_data = config('ext_data')
        super().__init__(str(self._dataset), self._userdata, *self._ext_data, backend='dataframe')

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

        self.make_index()

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
        self._check('elements')
        return self.dataset_elements

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
        self._check('elements_completed')
        return self.dataset_elements_completed


preset = Preset()
