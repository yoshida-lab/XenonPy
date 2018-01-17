# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

try:
    from pathlib import Path

    Path().expanduser()
except (ImportError, AttributeError):
    from pathlib2 import Path

from .__version__ import __version__ as version
from .decorator import *
from .descriptor import *
from .model import *
from .preprocess import *
from .visualization import *

PACKAGE_CONF_DIR = '.XenonPy'
PACKAGE = 'XenonPy'


def get_version():
    """
    Get package version.

    Returns
    -------
    str
        package version.
    """
    return version


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
    home = Path.home()
    dir_ = home / PACKAGE_CONF_DIR
    cfg_file = dir_ / 'conf.yml'
    with open(cfg_file) as f:
        conf = yaml.load(f)
    if not key:
        return conf
    else:
        return conf[key]


def _init_cfg_file(force=False):
    """
    create config file is not exist at ~/.XenonPy/conf.yml

    ..warning::
        Set **force=True** will reset all which under the `~/.XenonPy`` dir.

    Agrs
    ----
    force: bool
        force reset ``conf.yml`` to default and empty all dirs under ``~/.XenonPy``.
    """
    from shutil import rmtree, copyfile
    home = Path.home()
    dir_ = home / PACKAGE_CONF_DIR
    cfg_file = dir_ / 'conf.yml'

    dataset_dir = dir_ / 'dataset'
    cached_dir = dir_ / 'cached'

    if force:
        rmtree(dir_)

    if not dir_.is_dir():
        # create config root dir
        dir_.mkdir()

        # create other dirs
        dataset_dir.mkdir()
        cached_dir.mkdir()

    if not cfg_file.exists() or force:
        # copy default conf.yml to ~/.XenonPy
        copyfile(str(Path(__file__).parent / 'conf.yml'), cfg_file)


def _get_binary_file_url(name: str):
    """
    Return url with the given file name.

    Args
    ----
    name: str
        binary file name.

    Return
    ------
    str
        binary file url.
    """

    return 'https://github.com/TsumiNa/' + \
           PACKAGE + '/releases/download/v' + \
           get_version() + '/' + \
           name + '.pkl'


_init_cfg_file()
