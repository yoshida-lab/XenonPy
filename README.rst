.. Copyright 2017 TsumiNa. All rights reserved.


What is XenonPy project
========================
.. image:: https://badges.gitter.im/yoshida-lab/XenonPy.svg
    :alt: Join the chat at https://gitter.im/yoshida-lab/XenonPy
    :target: https://gitter.im/yoshida-lab/XenonPy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://travis-ci.org/yoshida-lab/XenonPy.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/yoshida-lab/XenonPy

.. image:: https://codecov.io/gh/yoshida-lab/XenonPy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/yoshida-lab/XenonPy

.. image:: https://img.shields.io/github/tag/yoshida-lab/XenonPy.svg?maxAge=360
    :alt: Version
    :target: https://github.com/yoshida-lab/XenonPy/releases/latest

.. image:: https://img.shields.io/pypi/pyversions/xenonpy.svg
    :alt: Python Versions
    :target: https://pypi.org/project/xenonpy/

**XenonPy** is a Python library that implements a comprehensive set of machine learning tools
for materials informatics. Its functionalities partially depend on PyTorch and R.
The current release (v0.2.0, 2018/12/25) is just a prototype version, which provides some limited modules:

* Interface to public materials database
* Library of materials descriptors (compositional/structural descriptors)
* Pretrained model library **XenonPy.MDL** (v0.1.0b, 2018/12/25: more than 10,000 modles in 35 properties of small molecules, polymers, and inorganic compounds)
* Machine learning tools.
* Transfer learning using the pretrained models in XenonPy.MDL

XenonPy inspired by matminer: https://hackingmaterials.github.io/matminer/.

XenonPy is a open source project https://github.com/yoshida-lab/XenonPy.

See our documents for details: http://xenonpy.readthedocs.io


Contribution guidelines
=======================

1. Fork it ( https://github.com/yoshida-lab/XenonPy/fork )
2. Create your feature branch (git checkout -b my-new-feature)
3. Commit your changes (git commit -am 'Add some feature')
4. Push to the branch (git push origin my-new-feature)
5. Create a new Pull Request

When contribute your codes, please do the following

* Discussion with others
* Docstring use `Numpy style`_.
* Check codes with Pylint_
* Writing tests if possible


Changes
=======

.. include: docs/source/changes.rst

Contact Us
==========

* With issues_
* With Gitter_


Copyright and license
=====================

©Copyright 2018 The XenonPy task force, all rights reserved.
Released under the `BSD-3 license`_.

.. _issues: https://github.com/yoshida-lab/XenonPy/issues
.. _BSD-3 license: https://opensource.org/licenses/BSD-3-Clause
.. _Gitter: https://gitter.im/yoshida-lab/XenonPy
.. _Numpy style: https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
.. _Pylint: https://pylint.readthedocs.io/
