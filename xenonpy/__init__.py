# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

__version__ = '0.1.0'
__release__ = 'b8'
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
    with open(str(cfg_file)) as f:
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
    root_dir = Path.home() / cfg_root
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
    userdata_dir = Path(get_conf('userdata')).expanduser()
    usermodel_dir = Path(get_conf('usermodel')).expanduser()

    # create dirs
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cached_dir.mkdir(parents=True, exist_ok=True)
    userdata_dir.mkdir(parents=True, exist_ok=True)
    usermodel_dir.mkdir(parents=True, exist_ok=True)


_init_cfg_file()

from . import descriptor
from . import model
# from .pipeline import *
# from .preprocess import *
from . import utils
from . import visualization
