# Copyright 2018 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from contextlib import contextmanager
from os import getenv
from pathlib import Path

from ruamel.yaml import YAML

from .._conf import __cfg_root__, __github_username__, __db_version__


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
        with open(str(Path(__file__).parents[1] / 'conf.yml')) as f:
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
    return 'https://github.com/' + __github_username__ + \
           '/dataset/releases/download/v' + __db_version__ + '/' + name + '.pkl.pd_'


def get_data_loc(name):
    """Return user data location"""

    scheme = ('userdata', 'usermodel')
    if name not in scheme:
        raise ValueError('{} not in {}'.format(name, scheme))
    if getenv(name):
        return str(Path(getenv(name)).expanduser())
    return str(Path(get_conf(name)).expanduser())


def absolute_path(path, ignore_err=True):
    """
    Resolve path when path include ``~``, ``parent/here``.

    Parameters
    ----------
    path: str
        Path to expand.
    ignore_err: bool
        FileNotFoundError is raised when set to False.
        When True, the path will be created.
    Returns
    -------
    str
        Expanded path.
    """
    from sys import version_info

    if isinstance(path, str):
        path = Path(path)

    if version_info[1] == 5:
        if ignore_err:
            path.expanduser().mkdir(parents=True, exist_ok=True)
        return str(path.expanduser().resolve())
    return str(path.expanduser().resolve(not ignore_err))


def get_sha256(fname):
    """
    Calculate file's sha256 value

    Parameters
    ----------
    fname: str
        File name.

    Returns
    -------
    str
        sha256 value.
    """
    from hashlib import sha256
    hasher = sha256()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class Switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        return

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
