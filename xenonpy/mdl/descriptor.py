#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from xenonpy.mdl.base import BaseQuery


class QueryDescriptorsWith(BaseQuery):
    queryable = [
        'name',
        'fullName',
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
            ) {{
                queryDescriptorsWith(
                    name_has: $name_has
                    fullName_has: $fullName_has
                    describe_has: $describe_has
                ) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class QueryDescriptors(BaseQuery):
    queryable = [
        'name',
        'fullName',
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
                queryDescriptors(query: $query) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class GetDescriptorDetail(BaseQuery):
    queryable = [
        'name',
        'fullName',
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
                getDescriptorDetail(name: $name) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class ListDescriptors(BaseQuery):
    queryable = [
        'name',
        'fullName',
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
                listDescriptors {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class CreateDescriptor(BaseQuery):
    queryable = [
        'name',
        'fullName',
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
            mutation ($with_: CreateDescriptorInput!) {{
                createDescriptor (with_: $with_) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''


class UpdateDescriptor(BaseQuery):
    queryable = [
        'name',
        'fullName',
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
                $with_: UpdateDescriptorInput!
            ) {{
                updateDescriptor (
                    name: $name
                    with_: $with_
                ) {{
                    {' '.join(query_vars)}
                }}
            }}
            '''
