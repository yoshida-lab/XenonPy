# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


from . import __github_username__, __cfg_root__


def get_conf(key: str):
    """
    Return config value with key or all config.

    Parameters
    ----------
    key: str
        Key of config item.

    Returns
    -------
    object
        key value in ``conf.yml`` file.
    """
    import yaml
    from pathlib import Path
    home = Path.home()
    dir_ = home / __cfg_root__
    cfg_file = dir_ / 'conf.yml'

    # from user local
    with open(str(cfg_file)) as f:
        conf = yaml.load(f)

    # if no key locally, use default
    if key not in conf:
        with open(str(Path(__file__).parent / 'conf.yml')) as f:
            conf_ = yaml.load(f)
            conf[key] = conf_[key]
        with open(str(cfg_file), 'w') as f:
            yaml.dump(conf, f)

    return conf[key]


def get_dataset_url(fname: str):
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


def init_cfg_file(force=False):
    """
    Create config file is not exist at ~/.xenonpy/conf.yml

    ..warning::
        Set **force=True** will reset all which under the `~/.xenonpy`` dir.

    Args
    ----
    force: bool
        force reset ``conf.yml`` to default and empty all dirs under ``~/.xenonpy``.
    """
    from shutil import rmtree, copyfile
    from pathlib import Path
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
    userdata_dir = Path(get_conf('userdata')).expanduser()
    usermodel_dir = Path(get_conf('usermodel')).expanduser()

    # create dirs
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cached_dir.mkdir(parents=True, exist_ok=True)
    userdata_dir.mkdir(parents=True, exist_ok=True)
    usermodel_dir.mkdir(parents=True, exist_ok=True)
