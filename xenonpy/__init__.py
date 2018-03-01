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

__all__ = ['descriptor', 'model', 'utils', 'visualization']

from . import descriptor
from . import model
# from .pipeline import *
# from .preprocess import *
from . import utils
from . import visualization


def _init_cfg_file(force=False):
    """
    Create config file is not exist at ~/.xenonpy/conf.yml

    .. warning::

        Set ``force=True`` will reset all which under the ``~/.xenonpy`` dir.

    Args
    ----
    force: bool
        force reset ``conf.yml`` to default and empty all dirs under ``~/.xenonpy``.
    """
    from shutil import rmtree, copyfile
    from pathlib import Path
    from ._conf import get_conf

    root_dir = Path.home() / __cfg_root__
    root_dir.mkdir(parents=True, exist_ok=True)
    cfg_file = root_dir / 'conf.yml'

    # copy default conf.yml to ~/.xenonpy
    if not cfg_file.exists() or force:
        copyfile(str(Path(__file__).parent / 'conf.yml'), str(cfg_file))

    if force:
        rmtree(str(root_dir))

    # init dirs
    dataset_dir = root_dir / 'dataset'
    cached_dir = root_dir / 'cached'
    user_data_dir = Path(get_conf('userdata')).expanduser()
    user_model_dir = Path(get_conf('usermodel')).expanduser()

    # create dirs
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cached_dir.mkdir(parents=True, exist_ok=True)
    user_data_dir.mkdir(parents=True, exist_ok=True)
    user_model_dir.mkdir(parents=True, exist_ok=True)
