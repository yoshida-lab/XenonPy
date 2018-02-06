# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

__version__ = '0.1.0'
__release__ = 'b9'
__short_description__ = "material descriptor library"
__license__ = "BSD (3-clause)"
__author__ = "TsumiNa"
__author_email__ = "liu.chang.1865@gmail.com"
__maintainer__ = "TsumiNa"
__maintainer_email__ = "liu.chang.1865@gmail.com"
__github_username__ = "yoshida-lab"
__cfg_root__ = '.' + __name__

__all__ = ['descriptor', 'model', 'utils', 'visualization', 'get_conf', 'get_dataset_url', 'init_cfg_file']

from . import descriptor
from . import model
# from .pipeline import *
# from .preprocess import *
from . import utils
from . import visualization

from .conf import init_cfg_file, get_conf, get_dataset_url

init_cfg_file()
