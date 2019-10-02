#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pandas as pd

from xenonpy.mdl.base import BaseQuery


class QueryModelDetailsWith(BaseQuery):
    common = [
        'id',
        'transferred',
        'succeed',
        'isRegression',
        'deprecated',
        'modelset',
        'method',
        'property',
        'descriptor',
        'lang',

    ]

    classification = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'sensitivity',
        'prevalence',
        'specificity',
        'ppv',
        'npv',
    ]

    regression = [
        'meanAbsError',
        'maxAbsError',
        'meanSquareError',
        'rootMeanSquareError',
        'r2',
        'pValue',
        'spearmanCorr',
        'pearsonCorr',
    ]

    queryable = common + classification + regression

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        reg, cls = [], []
        if 'id' not in query_vars:
            query_vars = query_vars + ('id',)
        for var in query_vars:
            if var in self.common:
                # common.append(var)
                reg.append(var)
                cls.append(var)
            elif var in self.regression:
                reg.append(var)
            elif var in self.classification:
                cls.append(var)

        return f'''
            query (
                $modelset_has: [String!]
                $property_has: [String!]
                $descriptor_has: [String!]
                $method_has: [String!]
                $lang_has: [String!]
                $regression: Boolean
                $transferred: Boolean
                $deprecated: Boolean
                $succeed: Boolean
            ) {{
                queryModelDetailsWith(
                    modelset_has: $modelset_has
                    property_has: $property_has
                    descriptor_has: $descriptor_has
                    method_has: $method_has
                    lang_has: $lang_has
                    regression: $regression
                    transferred: $transferred
                    deprecated: $deprecated
                    succeed: $succeed
                ) {{
                    ...on Regression {{
                        {' '.join(reg)}
                    }}
                    ...on Classification {{
                        {' '.join(cls)}
                    }}
                }}
            }}
            '''


class QueryModelDetails(BaseQuery):
    common = [
        'id',
        'transferred',
        'succeed',
        'isRegression',
        'deprecated',
        'modelset',
        'method',
        'property',
        'descriptor',
        'lang',

    ]

    classification = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'sensitivity',
        'prevalence',
        'specificity',
        'ppv',
        'npv',
    ]

    regression = [
        'meanAbsError',
        'maxAbsError',
        'meanSquareError',
        'rootMeanSquareError',
        'r2',
        'pValue',
        'spearmanCorr',
        'pearsonCorr',
    ]
    queryable = common + classification + regression

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        reg, cls = [], []
        if 'id' not in query_vars:
            query_vars = query_vars + ('id',)
        for var in query_vars:
            if var in self.common:
                reg.append(var)
                cls.append(var)
            elif var in self.regression:
                reg.append(var)
            elif var in self.classification:
                cls.append(var)

        return f'''
            query ($query: [String!]!) {{
                queryModelDetails(query: $query) {{
                    ...on Regression {{
                        {' '.join(reg)}
                    }}
                    ...on Classification {{
                        {' '.join(cls)}
                    }}
                }}
            }}
            '''


class GetModelUrls(BaseQuery):
    queryable = [
        'id',
        'etag',
        'url',
    ]

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        return f'''
            query ($ids: [Int!]!) {{
                getModelUrls(ids: $ids) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class GetModelUrl(BaseQuery):
    queryable = [
        'id',
        'etag',
        'url',
    ]

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        return f'''
            query ($id: Int!) {{
                getModelUrl(id: $id) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class GetModelDetails(BaseQuery):
    common = [
        'id',
        'transferred',
        'succeed',
        'isRegression',
        'deprecated',
        'modelset',
        'method',
        'property',
        'descriptor',
        'lang',

    ]

    classification = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'sensitivity',
        'prevalence',
        'specificity',
        'ppv',
        'npv',
    ]

    regression = [
        'meanAbsError',
        'maxAbsError',
        'meanSquareError',
        'rootMeanSquareError',
        'r2',
        'pValue',
        'spearmanCorr',
        'pearsonCorr',
    ]
    queryable = common + classification + regression

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        reg, cls = [], []
        for var in query_vars:
            if var in self.common:
                reg.append(var)
                cls.append(var)
            elif var in self.regression:
                reg.append(var)
            elif var in self.classification:
                cls.append(var)

        return f'''
            query ($ids: [Int!]!) {{
                getModelDetails(ids: $ids) {{
                    ...on Regression {{
                        {' '.join(reg)}
                    }}
                    ...on Classification {{
                        {' '.join(cls)}
                    }}
                }}
            }}
            '''
        pass


class GetModelDetail(BaseQuery):
    common = [
        'id',
        'transferred',
        'succeed',
        'isRegression',
        'deprecated',
        'modelset',
        'method',
        'property',
        'descriptor',
        'lang',

    ]

    classification = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'sensitivity',
        'prevalence',
        'specificity',
        'ppv',
        'npv',
    ]

    regression = [
        'meanAbsError',
        'maxAbsError',
        'meanSquareError',
        'rootMeanSquareError',
        'r2',
        'pValue',
        'spearmanCorr',
        'pearsonCorr',
    ]
    queryable = common + classification + regression

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        reg, cls = [], []
        for var in query_vars:
            if var in self.common:
                reg.append(var)
                cls.append(var)
            elif var in self.regression:
                reg.append(var)
            elif var in self.classification:
                cls.append(var)

        return f'''
            query ($id: Int!) {{
                getModelDetail(id: $id) {{
                    ...on Regression {{
                        {' '.join(reg)}
                    }}
                    ...on Classification {{
                        {' '.join(cls)}
                    }}
                }}
            }}
            '''
        pass


class GetTrainingInfo(BaseQuery):
    queryable = []

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    @staticmethod
    def _post(ret, return_json):
        return pd.DataFrame(ret)

    def gql(self, *query_vars: str):
        return f'''
            query ($id: Int!) {{
                getTrainingInfo(id: $id) 
            }}
            '''


class GetTrainingEnv(BaseQuery):
    queryable = []

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)
        self._return_json = True

    def gql(self, *query_vars: str):
        return f'''
            query ($id: Int!) {{
                getTrainingEnv(id: $id) 
            }}
            '''


class GetSupplementary(BaseQuery):
    queryable = []

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)
        self._return_json = True

    def gql(self, *query_vars: str):
        return f'''
            query ($id: Int!) {{
                getSupplementary(id: $id) 
            }}
            '''


class ListModelsWithProperty(BaseQuery):
    queryable = [
        'id',
        'transferred',
        'succeed',
        'isRegression',
        'deprecated',
        'modelset',
        'method',
        'property',
        'descriptor',
        'lang'
    ]

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        return f'''
            query ($name: String!) {{
                listModelsWithProperty(name: $name) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class ListModelsWithModelset(BaseQuery):
    queryable = [
        'id',
        'transferred',
        'succeed',
        'isRegression',
        'deprecated',
        'modelset',
        'method',
        'property',
        'descriptor',
        'lang'
    ]

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        return f'''
            query ($name: String!) {{
                listModelsWithModelset(name: $name) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class ListModelsWithMethod(BaseQuery):
    queryable = [
        'id',
        'transferred',
        'succeed',
        'isRegression',
        'deprecated',
        'modelset',
        'method',
        'property',
        'descriptor',
        'lang'
    ]

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        return f'''
            query ($name: String!) {{
                listModelsWithMethod(name: $name) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class ListModelsWithDescriptor(BaseQuery):
    queryable = [
        'id',
        'transferred',
        'succeed',
        'isRegression',
        'deprecated',
        'modelset',
        'method',
        'property',
        'descriptor',
        'lang'
    ]

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        return f'''
            query ($name: String!) {{
                listModelsWithDescriptor(name: $name) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class UploadModel(BaseQuery):
    queryable = [
        'id',
        'etag',
        'path'
    ]

    def __init__(self, variables, *, api_key: str = 'anonymous.user.key',
                 endpoint: str = 'http://xenon.ism.ac.jp/api'):
        """
        Access to XenonPy.MDL library.

        Parameters
        ----------
        api_key
            Not implement yet.
        """
        super().__init__(variables=variables, api_key=api_key, endpoint=endpoint)

    def gql(self, *query_vars: str):
        return f"""
                mutation(
                    $id: Int!
                    $describe: UploadModelInput!
                    $model: Upload!
                    $training_env: Json
                    $training_info: Json
                    $supplementary: Json
                ) {{
                    uploadModel(
                        modelsetId: $id
                        model: $model
                        describe: $describe
                        training_env: $training_env
                        training_info: $training_info
                        supplementary: $supplementary
                    ) {{
                        {' '.join(query_vars)}
                    }}
                }}
                """
