# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

__version__ = '0.1.0'
__release__ = 'b7'
__short_description__ = "material descriptor library"
__license__ = "BSD (3-clause)"
__author__ = "TsumiNa"
__author_email__ = "liu.chang.1865@gmail.com"
__maintainer__ = "TsumiNa"
__maintainer_email__ = "liu.chang.1865@gmail.com"
__github_username__ = "yoshida-lab"

cfg_root = '.' + __name__


def get_conf(key: str = None):
    """
    Return config value with key or all config.

    Parameters
    ----------
    key: str or None
        name of config item.
        left `None` will return config entry as dict.

    Returns
    -------
    object
        key value in ``conf.yml`` file.
    """
    import yaml
    from pathlib import Path
    home = Path.home()
    dir_ = home / cfg_root
    cfg_file = dir_ / 'conf.yml'
    with open(cfg_file) as f:
        conf = yaml.load(f)
    if not key:
        return conf
    else:
        return conf[key]


def _get_dataset_url(fname: str):
    """
    Return url with the given file name.

    Args
    ----
    fname: str
        binary file name.

    Return
    ------
    str
        binary file url.
    """
    return 'https://github.com/' + __github_username__ + '/dataset/releases/download/v0.1' + '/' + fname + '.pkl'


def _init_cfg_file(force=False):
    """
    Create config file is not exist at ~/.xenonpy/conf.yml

    ..warning::
        Set **force=True** will reset all which under the `~/.xenonpy`` dir.

    Agrs
    ----
    force: bool
        force reset ``conf.yml`` to default and empty all dirs under ``~/.xenonpy``.
    """
    from shutil import rmtree, copyfile
    from pathlib import Path
    home = Path.home()
    dir_ = home / cfg_root
    cfg_file = dir_ / 'conf.yml'

    dataset_dir = dir_ / 'dataset'
    userdata_dir = dir_ / 'userdata'
    cached_dir = dir_ / 'cached'

    if force:
        rmtree(str(dir_))

    if not dir_.is_dir():
        # create config root dir
        dir_.mkdir()

        # create other dirs
        dataset_dir.mkdir()
        cached_dir.mkdir()
        userdata_dir.mkdir()

    if not cfg_file.exists() or force:
        # copy default conf.yml to ~/.xenonpy
        copyfile(str(Path(__file__).parent / 'conf.yml'), str(cfg_file))


_init_cfg_file()

from . import descriptor
from . import model
# from .pipeline import *
# from .preprocess import *
from . import utils
from . import visualization
