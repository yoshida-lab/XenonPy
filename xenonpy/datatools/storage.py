#  Copyright (c) 2019. TsumiNa. All rights reserved.
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
from sklearn.externals import joblib

from ..utils import get_data_loc, absolute_path


class Storage(object):
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
        sub_set = Storage(name, path=str(self._path), backend=self._backend)
        setattr(self, name, sub_set)
        return sub_set

    def __repr__(self):
        cont_ls = ['<{}> under `{}` includes:'.format(self._name, self.path)]

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
