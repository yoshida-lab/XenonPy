# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


from contextlib import contextmanager
from os import getenv
from pathlib import Path

from ruamel.yaml import YAML

from .. import __cfg_root__


@contextmanager
def set_env(**kwargs):
    """
    Set temp environment variable with ``with`` statement.

    Examples
    --------
    >>> import os
    >>> with set_env(test='test env'):
    >>>    print(os.getenv('test'))
    test env
    >>> print(os.getenv('test'))
    None

    Parameters
    ----------
    kwargs: dict
        Dict with string value.
    """
    import os

    tmp = dict()
    for k, v in kwargs.items():
        tmp[k] = os.getenv(k)
        os.environ[k] = v
    yield
    for k, v in tmp.items():
        if not v:
            del os.environ[k]
        else:
            os.environ[k] = v


def __get_package_info(key):
    yaml = YAML(typ='safe')
    yaml.indent(mapping=2, sequence=4, offset=2)
    cwd = Path(__file__).parents[1] / 'conf.yml'
    with open(str(cwd), 'r') as f:
        info = yaml.load(f)
    return info[key]


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
    yaml = YAML(typ='safe')
    yaml.indent(mapping=2, sequence=4, offset=2)
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


def get_dataset_url(name: str):
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
    username = __get_package_info('github_username')
    return 'https://github.com/' + username + '/dataset/releases/download/v0.1.1' + '/' + name + '.pkl.pd_'


def get_data_loc(name):
    """Return user data location"""

    scheme = ('userdata', 'usermodel')
    if name not in scheme:
        raise ValueError('{} not in {}'.format(name, scheme))
    if getenv(name):
        return str(Path(getenv(name)).expanduser())
    return str(Path(get_conf(name)).expanduser())
