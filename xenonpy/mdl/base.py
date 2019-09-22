#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import json
from os import remove
from pathlib import Path
from shutil import make_archive

import pandas as pd
import requests
from requests import HTTPError
from sklearn.base import BaseEstimator

from xenonpy.utils import TimedMetaClass


class BaseQuery(BaseEstimator, metaclass=TimedMetaClass):
    queryable = None

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        if self.queryable is None:
            raise RuntimeError('Query class must give a queryable field in list of string')

        self._results = None
        self._return_json = False
        self._endpoint = endpoint
        self._api_key = api_key
        self._variables = variables

    @property
    def api_key(self):
        return self._api_key

    @property
    def endpoint(self):
        return self._endpoint

    @property
    def variables(self):
        return self._variables

    @property
    def results(self):
        return self._results

    def gql(self, *query_vars: str):
        raise NotImplementedError()

    @staticmethod
    def _post(ret, return_json):
        if return_json:
            return ret

        if not isinstance(ret, list):
            ret = [ret]
        ret = pd.DataFrame(ret)
        return ret

    def check_query_vars(self, *query_vars: str):
        if not set(query_vars) <= set(self.queryable):
            raise RuntimeError(f'`query_vars` contains illegal variables, '
                               f'available querying variables are: {self.queryable}')
        return query_vars

    def __call__(self, *querying_vars, file=None, return_json=None):
        if len(querying_vars) == 0:
            query = self.gql(*self.queryable)
        else:
            query = self.gql(*self.check_query_vars(*querying_vars))

        payload = json.dumps({'query': query, 'variables': self._variables})

        if file is None:
            ret = requests.post(url=self._endpoint,
                                data=payload,
                                headers={"content-type": "application/json",
                                         'api_key': self._api_key})
        else:
            file = Path(file).resolve()
            file = make_archive(str(file), 'gztar', str(file))
            operations = ('operations', payload)
            maps = ('map', json.dumps({0: ['variables.model']}))
            payload_tuples = (operations, maps)
            files = {'0': open(file, 'rb')}
            try:
                ret = requests.post(url=self._endpoint,
                                    data=payload_tuples,
                                    headers={'api_key': self._api_key},
                                    files=files)
            finally:
                files['0'].close()
                remove(file)

        if ret.status_code != 200:
            try:
                message = ret.json()
            except json.JSONDecodeError:
                message = "Server did not responce."

            raise HTTPError('status_code: %s, %s' %
                            (ret.status_code, message))
        ret = ret.json()
        if 'errors' in ret:
            raise ValueError(ret['errors'][0]['message'])
        query_name = self.__class__.__name__
        ret = ret['data'][query_name[0].lower() + query_name[1:]]

        if not ret:
            return None

        if return_json is None:
            return_json = self._return_json

        ret = self._post(ret, return_json)
        self._results = ret
        return ret

    def __repr__(self, N_CHAR_MAX=700):
        queryable = '\n '.join(self.queryable)
        return f'{super().__repr__(N_CHAR_MAX)}\nQueryable: \n {queryable}'
