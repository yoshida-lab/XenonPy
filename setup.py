# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-

# uncomment this for compatibility with py27
# from __future__ import print_function

import os
from pathlib import Path

from ruamel.yaml import YAML
from setuptools import setup, find_packages

# YOUR PACKAGE NAME
__package__ = 'xenonpy'


def get_requirements(filename: str):
    """
    Return requirements list from a text file.

    Parameters
    ----------
    filename: str
        Name of requirement file.

    Returns
    -------
        str-list
    """
    try:
        require = list()
        f = open(filename, "rb")
        for line in f.read().decode("utf-8").split("\n"):
            line = line.strip()
            if "#" in line:
                line = line[:line.find("#")].strip()
            if line:
                require.append(line)
    except IOError:
        print("'{}' not found!".format(filename))
        require = list()

    return require


class Package(object):
    def __init__(self):
        self.__info = None
        self.__name__ = __package__
        self.__version__ = self.get_info('version')
        self.__short_description__ = self.get_info('short_description')
        self.__license__ = self.get_info('license')
        self.__author__ = self.get_info('author')
        self.__author_email__ = self.get_info('author_email')
        self.__maintainer__ = self.get_info('maintainer')
        self.__maintainer_email__ = self.get_info('maintainer_email')
        self.__github_username__ = self.get_info('github_username')
        try:
            self.__long_description__ = open("README.rst", "rb").read().decode("utf-8")
        except FileNotFoundError:
            self.__long_description__ = "No long description!"

    def get_info(self, key):
        if not self.__info:
            yaml = YAML(typ='safe')
            yaml.indent(mapping=2, sequence=4, offset=2)
            cwd = Path(__file__).parent / __package__ / 'conf.yml'
            with open(str(cwd), 'r') as f:
                self.__info = yaml.load(f)
        try:
            return self.__info[key]
        except KeyError:
            return 'No <' + key + '>'

    def __getattr__(self, item: str):
        item = item.strip('_')
        return 'No <' + item + '>'


if __name__ == "__main__":
    # --- Automatically generate setup parameters ---
    package = Package()
    
    # Your package name
    PKG_NAME = package.__name__

    # Your GitHub user name
    GITHUB_USERNAME = package.__github_username__

    # Short description will be the description on PyPI
    SHORT_DESCRIPTION = package.__short_description__  # GitHub Short Description

    # Long description will be the body of content on PyPI page
    LONG_DESCRIPTION = package.__long_description__

    # Version number, VERY IMPORTANT!
    VERSION = package.__version__

    # Author and Maintainer
    AUTHOR = package.__author__

    # Author email
    AUTHOR_EMAIL = package.__author_email__

    MAINTAINER = package.__maintainer__

    MAINTAINER_EMAIL = package.__maintainer_email__

    PACKAGES, INCLUDE_PACKAGE_DATA, PACKAGE_DATA, PY_MODULES = (
        None,
        None,
        None,
        None,
    )

    # It's a directory style package
    if os.path.exists(__file__[:-8] + PKG_NAME):
        # Include all sub packages in package directory
        PACKAGES = find_packages(
            exclude=["*.tests", "*.tests.*", "tests.*", "tests"])

        # Include everything in package directory
        INCLUDE_PACKAGE_DATA = True
        PACKAGE_DATA = {
            "": ["*.*"],
        }

    # It's a single script style package
    elif os.path.exists(__file__[:-8] + PKG_NAME + ".py"):
        PY_MODULES = [
            PKG_NAME,
        ]

    # The project directory name is the GitHub repository name
    repository_name = package.__name__

    # Project Url
    URL = "https://github.com/{0}/{1}".format(GITHUB_USERNAME, repository_name)
    # Use todays date as GitHub release tag
    github_release_tag = 'v' + VERSION
    # Source code download url
    DOWNLOAD_URL = "https://github.com/{0}/{1}/archive/{2}.tar.gz".format(
        GITHUB_USERNAME, repository_name, github_release_tag)

    try:
        LICENSE = package.__license__
    except ImportError:
        print("'__license__' not found in '%s.__init__.py'!" % PKG_NAME)
        LICENSE = ""

    PLATFORMS = [
        "Windows",
        "MacOS",
        "Unix",
    ]

    CLASSIFIERS = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ]

    # Read requirements.txt, ignore comments
    INSTALL_REQUIRES = get_requirements("requirements.txt")
    SETUP_REQUIRES = ['pytest-runner', 'ruamel.yaml']
    TESTS_REQUIRE = ['pytest']
    setup(
        name=PKG_NAME,
        description=SHORT_DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        packages=PACKAGES,
        include_package_data=INCLUDE_PACKAGE_DATA,
        package_data=PACKAGE_DATA,
        py_modules=PY_MODULES,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=CLASSIFIERS,
        platforms=PLATFORMS,
        license=LICENSE,
        setup_requires=SETUP_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        tests_require=TESTS_REQUIRE)
"""
Appendix
--------
classifiers: https://pypi.python.org/pypi?%3Aaction=list_classifiers
"""
