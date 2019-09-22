#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from xenonpy.mdl.base import BaseQuery


class QueryPropertiesWith(BaseQuery):
    queryable = [
        'name',
        'fullName',
        'symbol',
        'unit',
        'describe',
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
                $fullName_has: [String!]
                $describe_has: [String!]
                $symbol_has: [String!]
                $unit_has: [String!]
            ) {{
                queryPropertiesWith(
                    name_has: $name_has
                    fullName_has: $fullName_has
                    describe_has: $describe_has
                    symbol_has: $symbol_has
                    unit_has: $unit_has
                ) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class QueryProperties(BaseQuery):
    queryable = [
        'name',
        'fullName',
        'symbol',
        'unit',
        'describe',
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
                queryProperties(query: $query) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class GetPropertyDetail(BaseQuery):
    queryable = [
        'name',
        'fullName',
        'symbol',
        'unit',
        'describe',
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
            query ($name: String!) {{
                getPropertyDetail(name: $name) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class ListProperties(BaseQuery):
    queryable = [
        'name',
        'fullName',
        'symbol',
        'unit',
        'describe',
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
                listProperties {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class CreateProperty(BaseQuery):
    queryable = [
        'name',
        'fullName',
        'symbol',
        'unit',
        'describe',
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
            mutation ($with_: CreatePropertyInput!) {{
                createProperty (with_: $with_) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class UpdateProperty(BaseQuery):
    queryable = [
        'name',
        'fullName',
        'symbol',
        'unit',
        'describe',
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
                $name: String!
                $with_: UpdatePropertyInput!
            ) {{
                updateProperty (
                    name: $name
                    with_: $with_
                ) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''
