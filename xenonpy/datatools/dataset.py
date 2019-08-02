#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import os
import re
from collections import defaultdict
from pathlib import Path
from warnings import warn

import joblib
import pandas as pd
import requests

__all__ = ['Dataset']


class Dataset(object):
    __extension__ = dict(
        pandas=(r'.*\.pd($|\.\w+$)', pd.read_pickle),
        csv=(r'csv', pd.read_csv),
        excel=(r'(xlsx|xls)', pd.read_excel),
        pickle=(r'.*\.pkl($|\.\w+$)', joblib.load))

    __re__ = re.compile(r'[\s\-.]')

    def __init__(self, *paths, backend='pandas', prefix=None):
        self._backend = backend
        self._files = None

        if len(paths) == 0:
            self._paths = ('.',)
        else:
            self._paths = paths

        if not prefix:
            prefix = ()
        self._prefix = prefix

        self._make_index(prefix=prefix)

    def _make_index(self, *, prefix):

        def make(path_):
            def _nest(_f):
                f_ = _f
                return lambda s: s.__extension__[s._backend][1](f_)

            patten = re.compile(self.__extension__[self._backend][0])
            for f in os.listdir(str(path_)):
                if patten.search(f):
                    fn = f.split('.')[0]
                    fn = self.__re__.sub('_', fn)
                    fp = str(path_ / f)

                    # select data
                    parent = re.split(r'[\\/]', str(path_))[-1]
                    # parent = str(f.parent).split('/')[-1]
                    if parent in prefix:
                        fn = '_'.join([parent, fn])

                    if fn in self._files:
                        warn(
                            "file %s with name %s already bind to %s and will be ignored" %
                            (fp, fn, self._files[fn]), RuntimeWarning)
                    else:
                        self._files[fn] = fp
                        setattr(self.__class__, fn, property(_nest(fp)))

        self._files = defaultdict(str)
        for path in self._paths:
            path = Path(path).expanduser().absolute()
            if not path.exists():
                raise RuntimeError('%s not exists' % str(path))
            make(path)

    @classmethod
    def to(cls, obj, path, *, force_pkl=False):
        if isinstance(path, Path):
            path = str(path)
        if not force_pkl and isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_pickle(path)
        return joblib.dump(obj, path)

    @classmethod
    def from_http(cls, url, save_to, *, filename=None, chunk_size=256 * 1024, params=None,
                  **kwargs):
        """
        Get file object via a http request.

        Parameters
        ----------
        url: str
            The resource url.
        save_to: str
            The path of a dir to save the downloaded object into it.
        filename: str, optional
            Specific the file name when saving.
            Set to ``None`` (default) to use a inferred name from http header.
        chunk_size: int, optional
            Chunk size.
        params: any, optional
            Parameters will be passed to ``requests.get`` function.
            See Also: `requests <http://docs.python-requests.org/>`_
        kwargs: dict, optional
            Pass to ``requests.get`` function as the ``kwargs`` parameters.

        Returns
        -------
        str
            File path contains file name.
        """
        r = requests.get(url, params, **kwargs)
        r.raise_for_status()

        if not filename:
            if 'filename' in r.headers:
                filename = r.headers['filename']
            else:
                filename = url.split('/')[-1]

        if isinstance(save_to, str):
            save_to = Path(save_to)
        if not isinstance(save_to, Path) or not save_to.is_dir():
            raise RuntimeError('%s is not a legal path or not point to a dir' % save_to)

        file_ = str(save_to / filename)
        with open(file_, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

        return file_

    def __repr__(self):
        cont_ls = ['<{}> includes:'.format(self.__class__.__name__)]

        for k, v in self._files.items():
            cont_ls.append('"{}": {}'.format(k, v))

        return '\n'.join(cont_ls)

    @property
    def csv(self):
        return Dataset(*self._paths, backend='csv', prefix=self._prefix)

    @property
    def pandas(self):
        return Dataset(*self._paths, backend='pandas', prefix=self._prefix)

    @property
    def pickle(self):
        return Dataset(*self._paths, backend='pickle', prefix=self._prefix)

    @property
    def excel(self):
        return Dataset(*self._paths, backend='excel', prefix=self._prefix)

    def __call__(self, *args, **kwargs):
        return self.__extension__[self._backend][1](*args, **kwargs)

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
        if name in self.__extension__:
            return self.__class__(*self._paths, backend=name, prefix=self._prefix)
        raise AttributeError("'%s' object has no attribute '%s'" % (self.__class__.__name__, name))
