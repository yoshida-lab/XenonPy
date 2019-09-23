#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from xenonpy.mdl.base import BaseQuery


class QueryModelsetsWith(BaseQuery):
    queryable = [
        'id',
        'name',
        'describe',
        'deprecated',
        'private',
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
            query (
                $name_has: [String!]
                $tag_has: [String!]
                $describe_has: [String!]
                $private: Boolean
                $deprecated: Boolean
            ) {{
                queryModelsetsWith(
                    name_has: $name_has
                    tag_has: $tag_has
                    describe_has: $describe_has
                    private: $private
                    deprecated: $deprecated
                ) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class QueryModelsets(BaseQuery):
    queryable = [
        'id',
        'name',
        'describe',
        'deprecated',
        'private',
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
            query ($query: [String!]!) {{
                queryModelsets(query: $query) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class GetModelsetDetail(BaseQuery):
    queryable = [
        'id',
        'name',
        'describe',
        'deprecated',
        'private',
        'contributors',
        'owner',
        'sampleCode',
        'tags',
        'count'
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
        self._return_json = True

    def gql(self, *query_vars: str):
        return f'''
            query ($id: Int!) {{
                getModelsetDetail(id: $id) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class ListModelsets(BaseQuery):
    queryable = [
        'id',
        'name',
        'describe',
        'deprecated',
        'private',
    ]

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

    def gql(self, *query_vars: str):
        return f'''
            query {{
                listModelsets {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class CreateModelset(BaseQuery):
    queryable = [
        'id',
        'name',
        'describe',
        'deprecated',
        'private',
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
        self._return_json = True

    def gql(self, *query_vars: str):
        return f'''
            mutation ($with_: CreateModelsetInput!) {{
                createModelset (with_: $with_) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class UpdateModelset(BaseQuery):
    queryable = [
        'id',
        'name',
        'describe',
        'deprecated',
        'private',
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
        self._return_json = True

    def gql(self, *query_vars: str):
        return f'''
            mutation (
                $id: Int!
                $with_: UpdateModelsetInput!
            ) {{
                updateModelset (
                    id: $id
                    with_: $with_
                ) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''
