#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from contextlib import contextmanager
from pathlib import Path

from ruamel.yaml import YAML

from xenonpy._conf import __cfg_root__


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
    kwargs: dict[str]
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


def config(key=None, **key_vals):
    """
    Return config value with key or all config.

    Parameters
    ----------
    key: str
        Keys of config item.
    key_vals: dict
        Set item's value by key.
    Returns
    -------
    str
        The value corresponding to the key.
    """
    yaml = YAML(typ='safe')
    yaml.indent(mapping=2, sequence=4, offset=2)
    dir_ = Path(__cfg_root__)
    cfg_file = dir_ / 'conf.yml'

    # from user local
    with open(str(cfg_file), 'r') as f:
        conf = yaml.load(f)

    value = None

    # getter
    if key:
        if key in conf:
            value = conf[key]
        else:
            tmp = Path(__file__).parent / 'conf.yml'
            with open(str(tmp)) as f:
                conf_ = yaml.load(f)

            if key not in conf_:
                raise RuntimeError('No item(s) named %s in configurations' % key)

            value = conf_[key]

    # setter  
    if key_vals:
        for key, v in key_vals.items():
            conf[key] = v
        with open(str(cfg_file), 'w') as f:
            yaml.dump(conf, f)

    return value
