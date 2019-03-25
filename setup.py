#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-

# uncomment this for compatibility with py27
# from __future__ import print_function

import os
from pathlib import Path

from ruamel.yaml import YAML
from setuptools import setup, find_packages

# YOUR PACKAGE NAME
__package__ = 'xenonpy'


class PackageInfo(object):
    def __init__(self, conf_file):
        yaml = YAML(typ='safe')
        yaml.indent(mapping=2, sequence=4, offset=2)
        with open(conf_file, 'r') as f:
            self._info = yaml.load(f)
        self.name = __package__

    @staticmethod
    def requirements(filename="requirements.txt"):
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

    def __getattr__(self, item: str):
        try:
            return self._info[item]
        except KeyError:
            return None


if __name__ == "__main__":
    # --- Automatically generate setup parameters ---
    cwd = Path(__file__).parent / __package__ / 'conf.yml'
    package = PackageInfo(str(cwd))

    # Your package name
    PKG_NAME = package.name

    # Your GitHub user name
    GITHUB_USERNAME = package.github_username

    # Short description will be the description on PyPI
    SHORT_DESCRIPTION = package.short_description  # GitHub Short Description

    # Long description will be the body of content on PyPI page
    LONG_DESCRIPTION = package.long_description

    # Version number, VERY IMPORTANT!
    VERSION = package.version + package.release

    # Author and Maintainer
    AUTHOR = package.author

    # Author email
    AUTHOR_EMAIL = package.author_email

    MAINTAINER = package.maintainer

    MAINTAINER_EMAIL = package.maintainer_email

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

    # Project Url
    GITHUB_URL = "https://github.com/{0}/{1}".format(GITHUB_USERNAME, PKG_NAME)
    # Use todays date as GitHub release tag
    RELEASE_TAG = 'v' + VERSION
    # Source code download url
    DOWNLOAD_URL = "https://github.com/{0}/{1}/archive/{2}.tar.gz".format(
        GITHUB_USERNAME, PKG_NAME, RELEASE_TAG)

    LICENSE = package.license or "'__license__' not found in '%s.__init__.py'!" % PKG_NAME

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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ]

    # Read requirements.txt, ignore comments
    INSTALL_REQUIRES = PackageInfo.requirements()
    SETUP_REQUIRES = ['pytest-runner', 'ruamel.yaml']
    TESTS_REQUIRE = ['pytest']
    setup(
        python_requires='~=3.6',
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
        url=GITHUB_URL,
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
