# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-

# uncomment this for compatibility with py27
# from __future__ import print_function

import os
from datetime import date

from setuptools import setup, find_packages

# --- import your package ---
import xenonpy as package


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


if __name__ == "__main__":
    # --- Automatically generate setup parameters ---
    # Your package name
    PKG_NAME = package.__name__

    # Your GitHub user name
    try:
        GITHUB_USERNAME = package.__github_username__
    except:
        GITHUB_USERNAME = "Unknown-Github-Username"

    # Short description will be the description on PyPI
    try:
        SHORT_DESCRIPTION = package.__short_description__  # GitHub Short Description
    except:
        print("'__short_description__' not found in '%s.__init__.py'!" %
              PKG_NAME)
        SHORT_DESCRIPTION = "No short description!"

    # Long description will be the body of content on PyPI page
    try:
        LONG_DESCRIPTION = open("README.rst", "rb").read().decode("utf-8")
    except:
        LONG_DESCRIPTION = "No long description!"

    # Version number, VERY IMPORTANT!
    VERSION = package.__version__ + package.__release__

    # Author and Maintainer
    try:
        AUTHOR = package.__author__
    except:
        AUTHOR = "Unknown"

    try:
        AUTHOR_EMAIL = package.__author_email__
    except:
        AUTHOR_EMAIL = None

    try:
        MAINTAINER = package.__maintainer__
    except:
        MAINTAINER = "Unknown"

    try:
        MAINTAINER_EMAIL = package.__maintainer_email__
    except:
        MAINTAINER_EMAIL = None

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
    except:
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
    REQUIRES = get_requirements("requirements.txt")
    REQUIRES_TEST = get_requirements("requirements_test.txt")

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
        setup_requires=['pytest-runner'] + REQUIRES,
        install_requires=REQUIRES,
        tests_require=REQUIRES_TEST,
    )
"""
Appendix
--------
classifiers: https://pypi.python.org/pypi?%3Aaction=list_classifiers
"""
