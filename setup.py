# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# -*- coding: utf-8 -*-

import ast

from setuptools import setup, find_packages

package_name = 'XenonPy'

def get_version(verFile):
    with open(verFile) as f:
        src = f.read()
    module = ast.parse(src)
    for e in module.body:
        if isinstance(e, ast.Assign) and \
                len(e.targets) == 1 and \
                e.targets[0].id == '__version__' and \
                isinstance(e.value, ast.Str):
            return e.value.s
    raise RuntimeError('__version__ not found')


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    this_license = f.read()

setup(
    name=package_name,
    version=get_version(package_name + '/__version__.py'),
    description='A library to generate descriptors of elements',
    long_description=readme,
    author='TsumiNa(Chang Liu)',
    author_email='liu.chang.1865@gmail.com',
    url='https://github.com/yoshida-lab/XenonPy',
    license=this_license,
    packages=find_packages(exclude=('tests', 'docs'), install_requires=['numpy']))
