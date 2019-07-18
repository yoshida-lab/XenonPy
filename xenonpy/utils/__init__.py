#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


from .env import config, set_env
from .file_path_url import get_data_loc, get_dataset_url, get_sha256, absolute_path
from .useful_cls import Switch, Singleton, TimedMetaClass, Timer
