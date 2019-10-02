#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import os
import tarfile
from pathlib import Path
from typing import List, Union

import pandas as pd
import requests
from sklearn.base import BaseEstimator
from tqdm import tqdm

from xenonpy.utils import TimedMetaClass
from .base import BaseQuery
from .descriptor import QueryDescriptors, QueryDescriptorsWith, UpdateDescriptor, CreateDescriptor, ListDescriptors, \
    GetDescriptorDetail
from .method import QueryMethods, QueryMethodsWith, UpdateMethod, CreateMethod, ListMethods, GetMethodDetail
from .model import QueryModelDetails, QueryModelDetailsWith, UploadModel, GetTrainingInfo, GetTrainingEnv, \
    GetSupplementary, GetModelUrls, GetModelUrl, GetModelDetails, GetModelDetail, ListModelsWithProperty, \
    ListModelsWithModelset, ListModelsWithMethod, ListModelsWithDescriptor
from .modelset import QueryModelsets, QueryModelsetsWith, UpdateModelset, CreateModelset, ListModelsets, \
    GetModelsetDetail
from .property import QueryPropertiesWith, QueryProperties, UpdateProperty, CreateProperty, ListProperties, \
    GetPropertyDetail

__all__ = ['MDL', 'GetVersion', 'QueryModelsetsWith', 'QueryModelsets', 'QueryModelDetailsWith', 'QueryModelDetails',
           'UpdateModelset', 'UploadModel', 'GetModelsetDetail', 'GetModelDetail', 'GetModelDetails', 'GetModelUrls',
           'GetModelUrl', 'GetTrainingInfo', 'GetSupplementary', 'GetTrainingEnv', 'ListModelsets',
           'ListModelsWithDescriptor', 'ListModelsWithMethod', 'ListModelsWithModelset', 'ListModelsWithProperty',
           'QueryPropertiesWith', 'QueryProperties', 'GetPropertyDetail', 'ListProperties', 'CreateProperty',
           'CreateModelset', 'UpdateProperty', 'QueryDescriptorsWith', 'QueryDescriptors', 'QueryMethodsWith',
           'QueryMethods', 'UpdateDescriptor', 'UpdateMethod', 'ListDescriptors', 'ListMethods', 'GetMethodDetail',
           'GetDescriptorDetail', 'CreateDescriptor', 'CreateMethod']


class GetVersion(BaseQuery):
    queryable = []

    def __init__(self, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables={}, api_key=api_key, endpoint=endpoint)
        self._return_json = True

    def gql(self, *query_vars: str):
        return '''
            query {
                getVersion 
            }
            '''


class MDL(BaseEstimator, metaclass=TimedMetaClass):
    def __init__(self, *, api_key: str = 'anonymous.user.key', endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Access key.
        endpoint:
            Url to XenonPy.MDL api server.
        """
        self._endpoint = endpoint
        self._api_key = api_key

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, e):
        self._endpoint = e

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, k):
        """"""
        self._api_key = k

    @property
    def version(self):
        return GetVersion()()

    def __call__(self, *query: str,
                 modelset_has: Union[List[str]] = None,
                 property_has: Union[List[str]] = None,
                 descriptor_has: Union[List[str]] = None,
                 method_has: Union[List[str]] = None,
                 lang_has: Union[List[str]] = None,
                 regression: bool = None,
                 transferred: bool = None,
                 deprecated: bool = None,
                 succeed: bool = None,
                 ) -> Union[QueryModelDetails, QueryModelDetailsWith]:
        """
        Query models with specific keywords.

        Parameters
        ----------
        query
            Lowercase string for database querying.
            This is a fuzzy searching, any information contains given string will hit.
        modelset_has
            The part of a model set's name.
            For example, ``modelset_has='test`` will hit ``*test*``
        property_has
            A part of the name of property.
        descriptor_has
            A part of the name of descriptor.
        method_has
            A part of the name of training method.
        lang_has
            A part of the name of programming language.
        regression
            If``True``, searching in regression models,
            else, searching in classification models.
            Default is ``True``.
        deprecated
            Model with this mark is deprecated.
        transferred
            If ``True``, searching in transferred models.
            Default is ``False``.
        succeed
            If ``True``, searching in succeed models.
            Default is ``True``.

        Returns
        -------
        query
            Querying object.
        """

        if len(query) > 0:
            return QueryModelDetails(dict(query=query), api_key=self.api_key, endpoint=self.endpoint)
        else:
            variables = dict(
                modelset_has=modelset_has,
                property_has=property_has,
                descriptor_has=descriptor_has,
                method_has=method_has,
                lang_has=lang_has,
                regression=regression,
                transferred=transferred,
                deprecated=deprecated,
                succeed=succeed
            )
            variables = {k: v for k, v in variables.items() if v is not None}

            return QueryModelDetailsWith(variables, api_key=self.api_key, endpoint=self.endpoint)

    def upload_model(self, *,
                     modelset_id: int,
                     describe: dict,
                     training_env: dict = None,
                     training_info: dict = None,
                     supplementary: dict = None) -> UploadModel:
        """
        Upload model to XenonPy.MDL server.

        Parameters
        ----------
        modelset_id
        describe
        training_env
        training_info
        supplementary

        Returns
        -------
        query
            Querying object.

        """
        variables = dict(
            model=None,
            id=modelset_id,
            describe=describe,
            training_env=training_env,
            training_info=training_info,
            supplementary=supplementary,
        )
        variables = {k: v for k, v in variables.items() if v is not None}

        return UploadModel(variables, api_key=self.api_key, endpoint=self.endpoint)

    def get_training_info(self, model_id: int) -> GetTrainingInfo:
        """
        Get training information, e.g. ``train_loss``.

        Parameters
        ----------
        model_id
            Model id.

        Returns
        -------
        query
            Querying object.
        """

        return GetTrainingInfo({'id': model_id}, api_key=self.api_key, endpoint=self.endpoint)

    def get_training_env(self, model_id: int) -> GetTrainingEnv:
        """
        Get model's training environments.

        Parameters
        ----------
        model_id
            Model id.

        Returns
        -------
        query
            Querying object.
        """
        return GetTrainingEnv({'id': model_id}, api_key=self.api_key, endpoint=self.endpoint)

    def get_supplementary(self, *, model_id: int) -> GetSupplementary:
        """
        Get extract information about the model.
        For regression model this should be *prediction vs observation* data.
        For classification model, this will be the *confusion matrix*.

        Parameters
        ----------
        model_id
            Model id.

        Returns
        -------
        query
            Querying object.
        """
        return GetSupplementary({'id': model_id}, api_key=self.api_key, endpoint=self.endpoint)

    def get_model_urls(self, *model_ids: int) -> Union[GetModelUrl, GetModelUrls]:
        """
        Get download url by model id.

        Parameters
        ----------
        model_ids
            Model id.

        Returns
        -------
        query
            Querying object.
        """
        if len(model_ids) == 0:
            raise RuntimeError('input is not non-able')
        if len(model_ids) == 1:
            return GetModelUrl({'id': model_ids[0]}, api_key=self.api_key, endpoint=self.endpoint)
        else:
            return GetModelUrls({'ids': model_ids}, api_key=self.api_key, endpoint=self.endpoint)

    def get_model_detail(self, model_id: int) -> GetModelDetail:
        """
        Get model detail by model id.

        Parameters
        ----------
        model_id
            Model id.

        Returns
        -------
        query
            Querying object.
        """
        return GetModelDetail({'id': model_id}, api_key=self.api_key, endpoint=self.endpoint)

    def get_model_details(self, model_ids: List[int]) -> GetModelDetails:
        """
        Get multiply model detail by their ids in one querying.

        Parameters
        ----------
        model_ids
            Model id.

        Returns
        -------
        query
            Querying object.
        """
        return GetModelDetails({'ids': model_ids}, api_key=self.api_key, endpoint=self.endpoint)

    def list_models_with_property(self, name: str) -> ListModelsWithProperty:
        """
        List all models relevant with given property.

        Parameters
        ----------
        name
            Property name

        Returns
        -------
        query
            Querying object.
        """
        return ListModelsWithProperty({'name': name}, api_key=self.api_key, endpoint=self.endpoint)

    def list_models_with_modelset(self, name: str) -> ListModelsWithModelset:
        """
        List all models relevant with given modelset.

        Parameters
        ----------
        name
            Modelset name

        Returns
        -------
        query
            Querying object.
        """
        return ListModelsWithModelset({'name': name}, api_key=self.api_key, endpoint=self.endpoint)

    def list_models_with_method(self, name: str) -> ListModelsWithMethod:
        """
        List all models relevant with given method.

        Parameters
        ----------
        name
            Method name

        Returns
        -------
        query
            Querying object.
        """
        return ListModelsWithMethod({'name': name}, api_key=self.api_key, endpoint=self.endpoint)

    def list_models_with_descriptor(self, name: str) -> ListModelsWithDescriptor:
        """
        List all models relevant with given descriptor.

        Parameters
        ----------
        name
            Descriptor name

        Returns
        -------
        query
            Querying object.
        """
        return ListModelsWithDescriptor({'name': name}, api_key=self.api_key, endpoint=self.endpoint)

    def query_modelsets(self, query: str = None, *,
                        name_has: Union[List[str]] = None,
                        tag_has: Union[List[str]] = None,
                        describe_has: Union[List[str]] = None,
                        private: bool = None,
                        deprecated: bool = None,
                        ) -> Union[QueryModelsets, QueryModelsetsWith]:
        """
        Query modelsets with specific keywords.

        Parameters
        ----------
        query
            Lowercase string for database querying.
            This is a fuzzy searching, any information contains given string will hit.
        name_has
            The part of a model set's name.
            For example, ``modelset_has='test`` will hit ``*test*``
        tag_has
            A part of the name of property.
        describe_has
            A part of the name of descriptor.
        private
            If``True``, searching in regression models,
            else, searching in classification models.
            Default is ``True``.
        deprecated
            Model with this mark is deprecated.
        deprecated
            If ``True``, searching in transferred models.
            Default is ``False``.

        Returns
        -------
        query
            Querying object.
        """

        if query is not None:
            return QueryModelsets(dict(query=query), api_key=self.api_key, endpoint=self.endpoint)
        else:
            variables = dict(
                name_has=name_has,
                tag_has=tag_has,
                describe_has=describe_has,
                private=private,
                deprecated=deprecated,
            )
            variables = {k: v for k, v in variables.items() if v is not None}

            return QueryModelsetsWith(variables, api_key=self.api_key, endpoint=self.endpoint)

    def update_modelset(self, *,
                        modelset_id: int,
                        name: str = None,
                        describe: str = None,
                        sample_code: str = None,
                        tags: List[str] = None,
                        private: bool = None,
                        deprecated: bool = None
                        ) -> UpdateModelset:
        """
        Update modelset information by modelset id.

        Parameters
        ----------
        modelset_id
        name
        describe
        sample_code
        tags
        private
        deprecated

        Returns
        -------
        query
            Querying object.
        """
        with_ = dict(
            name=name,
            describe=describe,
            sample_code=sample_code,
            tags=tags,
            private=private,
            deprecated=deprecated
        )
        with_ = {k: v for k, v in with_.items() if v is not None}

        return UpdateModelset({'id': modelset_id, 'with_': with_}, api_key=self.api_key, endpoint=self.endpoint)

    def create_modelset(self, *,
                        name: str,
                        describe: str = None,
                        sample_code: str = None,
                        tags: List[str] = None,
                        private: bool = False
                        ):
        """
        Create modelset.

        Parameters
        ----------
        name
        describe
        sample_code
        tags
        private

        Returns
        -------
        query
            Querying object.
        """
        with_ = dict(
            name=name,
            describe=describe,
            sample_code=sample_code,
            tags=tags,
            private=private
        )
        with_ = {k: v for k, v in with_.items() if v is not None}

        return CreateModelset({'with_': with_}, api_key=self.api_key, endpoint=self.endpoint)

    def list_modelsets(self) -> ListModelsets:
        """
        List all modelsets.

        Returns
        -------
        query
            Querying object.
        """
        return ListModelsets(api_key=self.api_key, endpoint=self.endpoint)

    def get_modelset_detail(self, modelset_id: int) -> GetModelsetDetail:
        """
        Get modelset detail by modelset id.

        Parameters
        ----------
        modelset_id

        Returns
        -------
        query
            Querying object.
        """
        return GetModelsetDetail({'id': modelset_id}, api_key=self.api_key, endpoint=self.endpoint)

    def query_descriptors(self, query: str = None, *,
                          name_has: Union[List[str]] = None,
                          fullname_has: Union[List[str]] = None,
                          describe_has: Union[List[str]] = None,
                          ) -> Union[QueryDescriptors, QueryDescriptorsWith]:
        """
        Query descriptors with specific keywords.

        Parameters
        ----------
        query
            Lowercase string for database querying.
            This is a fuzzy searching, any information contains given string will hit.
        name_has
            The part of a model set's name.
            For example, ``modelset_has='test`` will hit ``*test*``
        fullname_has
            Part of the full name.
        describe_has
            Part of the descriptor.

        Returns
        -------
        query
            Querying object.
        """

        if query is not None:
            return QueryDescriptors(dict(query=query), api_key=self.api_key, endpoint=self.endpoint)
        else:
            variables = dict(
                name_has=name_has,
                fullName_has=fullname_has,
                describe_has=describe_has,
            )
            variables = {k: v for k, v in variables.items() if v is not None}

            return QueryDescriptorsWith(variables, api_key=self.api_key, endpoint=self.endpoint)

    def update_descriptor(self, *,
                          name: str,
                          new_name: str = None,
                          describe: str = None,
                          fullname: str = None,
                          ) -> UpdateDescriptor:
        """
        Update descriptor information by name.

        Parameters
        ----------
        name
        new_name
        describe
        fullname

        Returns
        -------
        query
            Querying object.
        """

        with_ = dict(
            name=new_name,
            describe=describe,
            fullName=fullname,
        )
        with_ = {k: v for k, v in with_.items() if v is not None}

        return UpdateDescriptor({'name': name, 'with_': with_}, api_key=self.api_key, endpoint=self.endpoint)

    def create_descriptor(self, *,
                          name: str,
                          describe: str = None,
                          fullname: str = None,
                          ) -> CreateDescriptor:
        """
        Create descriptor.

        Parameters
        ----------
        name
        describe
        fullname

        Returns
        -------
        query
            Querying object.
        """

        with_ = dict(
            name=name,
            describe=describe,
            fullName=fullname,
        )
        with_ = {k: v for k, v in with_.items() if v is not None}

        return CreateDescriptor({'with_': with_}, api_key=self.api_key, endpoint=self.endpoint)

    def list_descriptors(self) -> ListDescriptors:
        """
        List all descriptor.

        Returns
        -------
        query
            Querying object.
        """
        return ListDescriptors(api_key=self.api_key, endpoint=self.endpoint)

    def get_descriptor_detail(self, name: str) -> GetDescriptorDetail:
        """
        Get descriptor detail by name.

        Parameters
        ----------
        name

        Returns
        -------
        query
            Querying object.
        """
        return GetDescriptorDetail({'name': name}, api_key=self.api_key, endpoint=self.endpoint)

    def query_methods(self, query: str = None, *,
                      name_has: Union[List[str]] = None,
                      fullname_has: Union[List[str]] = None,
                      describe_has: Union[List[str]] = None,
                      ) -> Union[QueryMethods, QueryMethodsWith]:
        """
        Query methods with specific keywords.

        Parameters
        ----------
        query
            Lowercase string for database querying.
            This is a fuzzy searching, any information contains given string will hit.
        name_has
            The part of a model set's name.
            For example, ``modelset_has='test`` will hit ``*test*``
        fullname_has
            Part of the full name.
        describe_has
            Part of the descriptor.

        Returns
        -------
        query
            Querying object.
        """

        if query is not None:
            return QueryMethods(dict(query=query), api_key=self.api_key, endpoint=self.endpoint)
        else:
            variables = dict(
                name_has=name_has,
                fullName_has=fullname_has,
                describe_has=describe_has,
            )
            variables = {k: v for k, v in variables.items() if v is not None}

            return QueryMethodsWith(variables, api_key=self.api_key, endpoint=self.endpoint)

    def update_method(self, *,
                      name: str,
                      new_name: str = None,
                      describe: str = None,
                      fullname: str = None,
                      ) -> UpdateMethod:
        """
        Update method information by name.

        Parameters
        ----------
        name
        new_name
        describe
        fullname

        Returns
        -------
        query
            Querying object.
        """

        with_ = dict(
            name=new_name,
            describe=describe,
            fullName=fullname,
        )
        with_ = {k: v for k, v in with_.items() if v is not None}

        return UpdateMethod({'name': name, 'with_': with_}, api_key=self.api_key, endpoint=self.endpoint)

    def create_method(self, *,
                      name: str,
                      describe: str = None,
                      fullname: str = None,
                      ) -> CreateMethod:
        """
        Create method.

        Parameters
        ----------
        name
        describe
        fullname

        Returns
        -------
        query
            Querying object.
        """

        with_ = dict(
            name=name,
            describe=describe,
            fullName=fullname,
        )
        with_ = {k: v for k, v in with_.items() if v is not None}

        return CreateMethod({'with_': with_}, api_key=self.api_key, endpoint=self.endpoint)

    def list_methods(self) -> ListMethods:
        """
        List all methods.

        Returns
        -------
        query
            Querying object.
        """
        return ListMethods(api_key=self.api_key, endpoint=self.endpoint)

    def get_method_detail(self, name: str) -> GetMethodDetail:
        """
        Get method detail by name.

        Parameters
        ----------
        name

        Returns
        -------
        query
            Querying object.
        """
        return GetMethodDetail({'name': name}, api_key=self.api_key, endpoint=self.endpoint)

    def query_properties(self, query: str = None, *,
                         name_has: Union[List[str]] = None,
                         fullname_has: Union[List[str]] = None,
                         describe_has: Union[List[str]] = None,
                         symbol_has: Union[List[str]] = None,
                         unit_has: Union[List[str]] = None,
                         ) -> Union[QueryProperties, QueryPropertiesWith]:
        """
        Query properties with specific keywords.

        Parameters
        ----------
        query
            Lowercase string for database querying.
            This is a fuzzy searching, any information contains given string will hit.
        name_has
            The part of a model set's name.
            For example, ``modelset_has='test`` will hit ``*test*``
        fullname_has
            Part of the full name.
        describe_has
            Part of the descriptor.
        symbol_has
            Part of the symbol.
        unit_has
            Part of the unit.

        Returns
        -------
        query
            Querying object.
        """

        if query is not None:
            return QueryProperties(dict(query=query), api_key=self.api_key, endpoint=self.endpoint)
        else:
            variables = dict(
                name_has=name_has,
                fullName_has=fullname_has,
                describe_has=describe_has,
                symbol_has=symbol_has,
                unit_has=unit_has,
            )
            variables = {k: v for k, v in variables.items() if v is not None}

            return QueryPropertiesWith(variables, api_key=self.api_key, endpoint=self.endpoint)

    def update_property(self, *,
                        name: str,
                        new_name: str = None,
                        describe: str = None,
                        fullname: str = None,
                        symbol: str = None,
                        unit: str = None,
                        ) -> UpdateProperty:
        """
        Update property information by name.

        Parameters
        ----------
        name
        new_name
        describe
        fullname
        symbol
        unit

        Returns
        -------
        query
            Querying object.
        """
        with_ = dict(
            name=new_name,
            describe=describe,
            fullName=fullname,
            symbol=symbol,
            priunitvate=unit,
        )
        with_ = {k: v for k, v in with_.items() if v is not None}

        return UpdateProperty({'name': name, 'with_': with_}, api_key=self.api_key, endpoint=self.endpoint)

    def create_property(self, *,
                        name: str,
                        describe: str = None,
                        fullname: str = None,
                        symbol: str = None,
                        unit: str = None,
                        ) -> CreateProperty:
        """
        Create property.

        Parameters
        ----------
        name
        describe
        fullname
        symbol
        unit

        Returns
        -------
        query
            Querying object.
        """

        with_ = dict(
            name=name,
            describe=describe,
            fullName=fullname,
            symbol=symbol,
            unit=unit
        )
        with_ = {k: v for k, v in with_.items() if v is not None}

        return CreateProperty({'with_': with_}, api_key=self.api_key, endpoint=self.endpoint)

    def list_properties(self) -> ListProperties:
        """
        List all properties.

        Returns
        -------
        query
            Querying object.
        """
        return ListProperties(api_key=self.api_key, endpoint=self.endpoint)

    def get_property_detail(self, name: str):
        """
        Get property detail by name.

        Parameters
        ----------
        name

        Returns
        -------
        query
            Querying object.
        """
        return GetPropertyDetail({'name': name}, api_key=self.api_key, endpoint=self.endpoint)

    def pull(self, *model_ids: Union[int, pd.Series, pd.DataFrame],
             save_to: str = '.') -> pd.DataFrame:
        """
        Download model(s) from XenonPy.MDL server.

        Parameters
        ----------
        model_ids
            Model ids.
            It can be given by a dataframe.
            In this case, the column with name ``id`` will be used.
        save_to
            Path to save models.

        Returns
        -------

        """
        if len(model_ids) == 0:
            raise RuntimeError('input can not be empty')
        if len(model_ids) == 1:
            if isinstance(model_ids[0], pd.Series):
                model_ids = model_ids[0].tolist()

            if isinstance(model_ids[0], pd.DataFrame):
                model_ids = model_ids[0]['id'].tolist()

        ret = self.get_model_urls(*model_ids)('id', 'url')
        urls = ret['url'].tolist()

        path_list = []
        for url in tqdm(urls):
            path = '/'.join(url.split('/')[4:])
            file = Path(save_to).resolve() / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file = str(file)
            r = requests.get(url=url)
            with open(file, 'wb') as f:
                f.write(r.content)
            path = file[:-7]
            tarfile.open(file).extractall(path=path)
            os.remove(file)
            path_list.append(path)

        ret['model'] = path_list
        return ret.drop(columns='url')
