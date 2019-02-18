#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# %%
import json

import requests


def get_properties(name):
    # url = 'http://xenon.ism.ac.jp/api'
    url = 'http://localhost:3001/api'
    headers = {'Accept': 'application/json', "content-type": "application/json"}
    query = '''
    query ($name: String!){
      queryProperties(name: $name){
        name
        models
      }
    }
    '''
    variables = {"name": name}
    payload = json.dumps({'query': query, 'variables': variables})

    ret = requests.post(url=url, headers=headers, data=payload)
    return ret
