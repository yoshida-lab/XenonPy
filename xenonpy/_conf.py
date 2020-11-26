#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from pathlib import Path

from ruamel.yaml import YAML

__all__ = [
    '__pkg_name__',
    '__version__',
    '__db_version__',
    '__release__',
    '__short_description__',
    '__license__',
    '__author__',
    '__author_email__',
    '__maintainer__',
    '__maintainer_email__',
    '__github_username__',
    '__cfg_root__',
]


class __PackageInfo(object):

    def __init__(self):
        yaml = YAML(typ='safe')
        yaml.indent(mapping=2, sequence=4, offset=2)
        cwd = Path(__file__).parent / 'conf.yml'
        with open(str(cwd), 'r') as f:
            info = yaml.load(f)
        self._info = info

    def __getattr__(self, item):
        try:
            return self._info[item]
        except KeyError:
            return None


package_info = __PackageInfo()

__pkg_name__ = package_info.name
__version__ = package_info.version
__db_version__ = package_info.db_version
__release__ = package_info.release
__short_description__ = package_info.short_description
__license__ = package_info.license
__author__ = package_info.author
__author_email__ = package_info.author_email
__maintainer__ = package_info.maintainer
__maintainer_email__ = package_info.maintainer_email
__github_username__ = package_info.github_username
__cfg_root__ = str(Path.home().resolve() / ('.' + __pkg_name__))
