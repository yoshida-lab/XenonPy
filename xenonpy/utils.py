#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import time
import types
from collections import defaultdict
from contextlib import contextmanager
from datetime import timedelta
from functools import wraps
from os import getenv
from pathlib import Path

from ruamel.yaml import YAML

from ._conf import __cfg_root__, __github_username__, __db_version__


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


def get_dataset_url(name, version=__db_version__):
    """
    Return url with the given file name.

    Args
    ----
    name: str
        binary file name.
    version: str
        The version of repository.
        See Also: https://github.com/yoshida-lab/dataset/releases

    Return
    ------
    str
        binary file url.
    """
    return 'https://github.com/{0}/dataset/releases/download/v{1}/{2}.pkl.pd_'.format(__github_username__, version,
                                                                                      name)


def get_data_loc(name):
    """Return user data location"""

    scheme = ('userdata', 'usermodel')
    if name not in scheme:
        raise ValueError('{} not in {}'.format(name, scheme))
    if getenv(name):
        return str(Path(getenv(name)).expanduser())
    return str(Path(config(name)).expanduser())


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


class Timer(object):
    class _Timer:
        def __init__(self):
            self.start = None
            self.times = []

        @property
        def elapsed(self):
            all_ = sum(self.times)
            dt = 0.0
            if self.start:
                dt = time.perf_counter() - self.start
            return all_ + dt

        def __repr__(self):
            return 'elapsed: %s' % str(timedelta(seconds=self.elapsed))

    def __init__(self, time_func=time.perf_counter):
        self._func = time_func
        self._timers = defaultdict(self._Timer)

    def __call__(self, *args, **kwargs):
        self.timed(*args, **kwargs)

    def start(self, fn_name='main'):
        if self._timers[fn_name].start is not None:
            raise RuntimeError('Timer <%s> Already started' % fn_name)
        self._timers[fn_name].start = self._func()

    def stop(self, fn_name='main'):
        if self._timers[fn_name].start is None:
            raise RuntimeError('Timer <%s> not started' % fn_name)
        elapsed = self._func() - self._timers[fn_name].start
        self._timers[fn_name].times.append(elapsed)
        self._timers[fn_name].start = None

    @property
    def elapsed(self):
        if 'main' in self._timers:
            return self._timers['main'].elapsed
        return sum([v.elapsed for v in self._timers.values()])

    def __repr__(self):
        return 'Total elapsed: %s\n' % str(timedelta(seconds=self.elapsed)) + \
               '\n'.join(['  |- %s: %s' % (k, v.elapsed) for k, v in self._timers.items()])

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class TimedMetaClass(type):
    """
    This metaclass replaces each methods of its classes
    with a new function that is timed
    """

    def _timed(fn):
        if isinstance(fn, (types.FunctionType, types.MethodType)):
            @wraps(fn)
            def fn_(self, *args, **kwargs):
                self._timer.start(fn.__name__)
                try:
                    rt = fn(self, *args, **kwargs)
                finally:
                    self._timer.stop(fn.__name__)
                return rt

            return fn_
        raise TypeError('Need <FunctionType> or <MethodType> but got %s' % type(fn))

    def __new__(mcs, name, bases, attrs):

        if '__init__' in attrs:
            real_init = attrs['__init__']

            # we do a deepcopy in case default is mutable
            # but beware, this might not always work
            @wraps(real_init)
            def injected_init(self, *args, **kwargs):
                setattr(self, '_timer', Timer())
                # call the "real" __init__ that we hid with our injected one
                real_init(self, *args, **kwargs)
        else:
            def injected_init(self):
                setattr(self, '_timer', Timer())
        # inject it
        attrs['__init__'] = injected_init

        for name_, value_ in attrs.items():
            if not name_.startswith("__") and isinstance(value_, (types.FunctionType, types.MethodType)):
                attrs[name_] = TimedMetaClass._timed(value_)

        return super(TimedMetaClass, mcs).__new__(mcs, name, bases, attrs)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
