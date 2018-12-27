.. Copyright 2017 TsumiNa. All rights reserved.


.. role:: raw-html(raw)
    :format: html

========================
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


Overview
========

**XenonPy** is a Python library that implements a comprehensive set of machine learning tools
for materials informatics. Its functionalities partially depend on PyTorch and R.
The current release (v0.2.0, 2018/12/25) is just a prototype version, which provides some limited modules:

* Interface to public materials database
* Library of materials descriptors (compositional/structural descriptors)
* Pretrained model library **XenonPy.MDL** (v0.1.0b, 2018/12/25: more than 10,000 models in 35 properties of small molecules, polymers, and inorganic compounds)
* Machine learning tools.
* Transfer learning using the pretrained models in XenonPy.MDL


.. image:: _static/xenonpy.png



Features
========

XenonPy has a rich set of tools for apply materials informatics rapidly.
The descriptor generator class can calculate several types of numeric descriptors from ``compositional``, ``structure``.
By using XenonPy built in visualization function. The relationships between descriptors and target properties can easily be shown in a heatmap.

We have a great confidence in transfer learning. To facilitate the widespread use of transfer learning,
we have developed a comprehensive library of pre-trained models, called **XenonPy.MDL**.
This library provides simple API to enable users to fetch the models via a http request.
To play with pre-trained models, some useful functions are also provided.

See :doc:`features` for details

Reference
=========

Yamada, H., Liu, C., Wu, S., Koyama, Y., Ju, S., Shiomi, J., Morikawa, J., Yoshida, R.
*Transfer learning: a key driver of accelerating materials discovery with machine learning*, in preparation.



Sample
======

Some samples available here: https://github.com/yoshida-lab/XenonPy/tree/master/samples



Contributing
============
XenonPy is an `open source project <https://github.com/yoshida-lab/XenonPy>`_ inspired by `matminer <https://hackingmaterials.github.io/matminer>`_.
This project still under hard working. We appreciate any feedback.
Code contributions are welcomed. See :doc:`contribution` for details.


.. _user-doc:
.. toctree::
    :maxdepth: 2
    :caption: User Documentation

    installation
    features
    dataset
    contribution
    changes
    contact

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Copyright and license
=====================
©Copyright 2018 The XenonPy task force, all rights reserved.
Released under the `BSD-3 license`_.

.. _pandas: https://pandas.pydata.org
.. _PyTorch: http://pytorch.org/
.. _BSD-3 license: https://opensource.org/licenses/BSD-3-Clause
.. _Xenon: https://en.wikipedia.org/wiki/Xenon
