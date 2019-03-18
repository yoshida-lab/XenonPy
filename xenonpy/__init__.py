#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# change version in there, conf.yml, setup.py
__all__ = ['descriptor', 'model', 'utils', 'visualization', 'datatools', 'math']

from ._conf import *


def __init(force=False):
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
    from ruamel.yaml import YAML
    from sys import version_info
    from pathlib import Path

    if version_info[0] != 3 or version_info[1] < 5:
        raise SystemError("Python version must be greater than or equal to 3.5")

    yaml = YAML(typ='safe')
    yaml.indent(mapping=2, sequence=4, offset=2)
    root_dir = Path(__cfg_root__)
    root_dir.mkdir(parents=True, exist_ok=True)
    user_cfg_file = root_dir / 'conf.yml'

    if force:
        rmtree(str(root_dir))

    # copy default conf.yml to ~/.xenonpy
    if not user_cfg_file.exists() or force:
        copyfile(str(Path(__file__).parent / 'conf.yml'), str(user_cfg_file))
    else:
        user_cfg = yaml.load(user_cfg_file)
        if 'version' not in user_cfg or user_cfg['version'] != __version__:
            with open(str(Path(__file__).parent / 'conf.yml'), 'r') as f:
                pack_cfg = yaml.load(f)
            pack_cfg['userdata'] = user_cfg['userdata']
            pack_cfg['usermodel'] = user_cfg['usermodel']
            yaml.dump(pack_cfg, user_cfg_file)

    # init dirs
    user_cfg = yaml.load(user_cfg_file)
    dataset_dir = root_dir / 'dataset'
    cached_dir = root_dir / 'cached'
    user_data_dir = Path(user_cfg['userdata']).expanduser().absolute()
    user_model_dir = Path(user_cfg['usermodel']).expanduser().absolute()

    # create dirs
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cached_dir.mkdir(parents=True, exist_ok=True)
    user_data_dir.mkdir(parents=True, exist_ok=True)
    user_model_dir.mkdir(parents=True, exist_ok=True)


__init()

from . import datatools
from . import descriptor
from . import math
from . import model
# from . import pipeline
from . import utils
from . import visualization
