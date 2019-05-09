.. Copyright 2017 TsumiNa. All rights reserved.


.. role:: raw-html(raw)
    :format: html

========================
What is XenonPy project
========================

.. image:: https://travis-ci.org/yoshida-lab/XenonPy.svg?branch=master
    :alt: Build Status
    :target: https://travis-ci.org/yoshida-lab/XenonPy

.. image:: https://ci.appveyor.com/api/projects/status/vnh350xqffp6t9nk/branch/master?svg=true
    :alt: Build Status
    :target: https://ci.appveyor.com/project/TsumiNa/xenonpy

.. image:: https://codecov.io/gh/yoshida-lab/XenonPy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/yoshida-lab/XenonPy

.. image:: https://img.shields.io/github/tag/yoshida-lab/XenonPy.svg?maxAge=360
    :alt: Version
    :target: https://github.com/yoshida-lab/XenonPy/releases/latest

.. image:: https://img.shields.io/pypi/pyversions/xenonpy.svg
    :alt: Python Versions
    :target: https://pypi.org/project/xenonpy/

.. image:: https://badges.gitter.im/yoshida-lab/XenonPy.svg
    :alt: Join the chat at https://gitter.im/yoshida-lab/XenonPy
    :target: https://gitter.im/yoshida-lab/XenonPy?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

--------
Overview
--------
**XenonPy** is a Python library that implements a comprehensive set of machine learning tools
for materials informatics. Its functionalities partially depend on Python (PyTorch) and R (MXNet).
The current release (v0.2.1, 2019/2/20) is a prototype version, which provides some limited modules:

* Interface to the public materials database
* Library of materials descriptors (compositional/structural descriptors)
* Pretrained model library **XenonPy.MDL** (v0.1.0b, 2018/12/25: more than 10,000 models in 35 properties of small molecules, polymers, and inorganic compounds)
* Machine learning tools.
* Transfer learning using the pre-trained models in XenonPy.MDL


.. image:: _static/xenonpy.png


--------
Features
--------
XenonPy has a rich set of tools for various materials informatics applications.
The descriptor generator class can calculate several types of numeric descriptors from ``compositional``, ``structure``.
By using XenonPy's built-in visualization functions, the relationships between descriptors and target properties can be easily shown in a heatmap.

Transfer learning is an important tool for the efficient application of machine learning methods to materials informatics.
To facilitate the widespread use of transfer learning,
we have developed a comprehensive library of pre-trained models, called **XenonPy.MDL**.
This library provides a simple API that allows users to fetch the models via an HTTP request.
For the ease of using the pre-trained models, some useful functions are also provided.

See :doc:`features` for details


------
Sample
------
Some samples are available here: https://github.com/yoshida-lab/XenonPy/tree/master/samples


.. _user-doc:
.. toctree::
    :maxdepth: 2
    :caption: User Documentation

    copyright
    installation
    features
    tutorial
    api
    contribution
    changes
    contact


---------
Reference
---------

Yamada, H., Liu, C., Wu, S., Koyama, Y., Ju, S., Shiomi, J., Morikawa, J., Yoshida, R.
*Transfer learning: a key driver of accelerating materials discovery with machine learning*, in preparation.


------------
Contributing
------------

XenonPy is an `open source project <https://github.com/yoshida-lab/XenonPy>`_ inspired by `matminer <https://hackingmaterials.github.io/matminer>`_.
:raw-html:`<br/>`
This project is under on-going development. We would appreciate any feedback from the users.
:raw-html:`<br/>`
Code contributions are also very welcomed. See :doc:`contribution` for more details.


.. _pandas: https://pandas.pydata.org
.. _PyTorch: http://pytorch.org/
.. _Xenon: https://en.wikipedia.org/wiki/Xenon
