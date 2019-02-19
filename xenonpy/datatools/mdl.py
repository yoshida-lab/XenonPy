#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# %%
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests import HTTPError
from sklearn.base import BaseEstimator
from tqdm import tqdm

from ..utils import TimedMetaClass, absolute_path


class MDL(BaseEstimator, metaclass=TimedMetaClass):
    def __init__(self, *, save_to='.', api_key=''):
        """
        Access to XenonPy.MDL.

        Parameters
        ----------
        save_to: str
            Path to save models.
            If set to ``None``, only return query result.
        api_key: str
            Not implement yet.
        """
        self.save_to = save_to
        self._url = 'http://xenon.ism.ac.jp/api'
        self.api_key = api_key
        self._headers = {'Accept': 'application/json', "content-type": "application/json"}

    def query(self, query, variables):
        payload = json.dumps({'query': query, 'variables': variables})
        ret = requests.post(url=self._url, headers=self._headers, data=payload)
        if ret.status_code != 200:
            raise HTTPError('status_code: %s, %s' %
                            (ret.status_code, ret.json()))
        ret = ret.json()['data']
        return ret

    def query_properties(self, name_has, *,
                         query_fullname=False,
                         query_symbol=False,
                         query_models=False,
                         query_unit=False):
        query_fullname = 'fullname' if query_fullname else ''
        query_symbol = 'symbol' if query_symbol else ''
        query_unit = 'unit' if query_unit else ''
        query_models = 'models' if query_models else ''
        query = '''
        query ($name: String!){
          queryProperties(name: $name){
            name
            %s
            %s
            %s
            %s
          }
        }
        ''' % (query_fullname, query_symbol, query_unit, query_models)

        variables = {"name": name_has}
        return self.query(query, variables)['queryProperties']

    def _query_models(self, modelset_has, *,
                      property_has='',
                      descriptor_has='',
                      method_has='',
                      lang_has='',
                      regress_is=True,
                      transferred_is=False,
                      succeed_is=True):
        query = '''
        query ($modelSet: String!
               $property: String
               $descriptor: String
               $method: String
               $lang: String
               $regress: Boolean
               $transferred: Boolean
               $succeed: Boolean)
        {
          queryModels(modelSet: $modelSet
               property: $property
               descriptor: $descriptor
               method: $method
               lang: $lang
               regress: $regress
               transferred: $transferred
               succeed: $succeed)
          {
            id
            url
            modelSet
            property
            descriptor
            method
            lang
            regress
            transferred
            succeed
            ...on Regression {
              mae
              r
            }
            ...on Classification {
              accuracy
            }
          }
        }
        '''

        variables = dict(
            modelSet=modelset_has,
            property=property_has,
            descriptor=descriptor_has,
            method=method_has,
            lang=lang_has,
            regress=regress_is,
            transferred=transferred_is,
            succeed=succeed_is
        )
        return self.query(query, variables)['queryModels']

    def __call__(self, modelset_has, *,
                 save_to='.',
                 property_has='',
                 descriptor_has='',
                 method_has='',
                 lang_has='',
                 regress_is=True,
                 transferred_is=False,
                 succeed_is=True):
        """
        Query models and download to a specific destination

        Parameters
        ----------
        modelset_has: str
            The part of a model set's name.
            For example, ``modelset_has='test`` will hit ``*test*``
        save_to: str or bool
            Path to save models.
            If ``False``, only return query results.
            This is a temporary change that only have effect in the current fetch.
        property_has: str
            A part of the name of property.
        descriptor_has: str
            A part of the name of descriptor.
        method_has: str
            A part of the name of training method.
        lang_has: str
            A part of the name of programming language.
        regress_is: bool
            If``True``, searching in regression models,
            else, searching in classification models.
            Default is ``True``.
        transferred_is: bool
            If ``True``, searching in transferred models.
            Default is ``False``.
        succeed_is: bool
            If ``True``, searching in succeed models.
            Default is ``True``.

        Returns
        -------
        ret: pd.DataFrame
            A summary of all downloaded models.
        """
        ret = self._query_models(modelset_has,
                                 property_has=property_has,
                                 descriptor_has=descriptor_has,
                                 method_has=method_has,
                                 lang_has=lang_has,
                                 regress_is=regress_is,
                                 transferred_is=transferred_is,
                                 succeed_is=succeed_is)
        if not ret:
            return None
        ret_ = pd.DataFrame(ret)
        if save_to is None:
            save_to = self.save_to
        if save_to:
            ret_['save_path'] = self.pull(ret_['url'], save_to=save_to)

        return ret_.set_index('id', drop=True)

    @classmethod
    def pull(cls, urls, save_to='.'):
        path_list = []
        if isinstance(urls, (np.ndarray, pd.Series)):
            urls = urls.tolist()
        for url in tqdm(urls):
            path = '/'.join(url.split('/')[-5:])
            filename = '{}/{}'.format(absolute_path(save_to), path)
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(url=url)
            with open(filename, 'wb') as f:
                f.write(r.content)
            path = filename[:-7]
            tarfile.open(filename).extractall(path=path)
            os.remove(filename)
            path_list.append(path)
        return path_list
