.. Copyright 2019 TsumiNa. All rights reserved.


.. role:: raw-html(raw)
    :format: html

========================
What is XenonPy project
========================

|MacOS| |Windows| |Ubuntu| |readthedocs| |codecov| |version| |python| |total-dl| |per-dl|

.. note::

   To all those who have purchased the book `マテリアルズインフォマティクス`_  published by `KYORITSU SHUPPAN`_:

   The link to the exercises has changed to https://github.com/yoshida-lab/XenonPy/tree/master/mi_book.
   Please follow the new link to access all these exercises.

   We apologize for the inconvenience.

--------
Overview
--------
**XenonPy** is a Python library that implements a comprehensive set of machine learning tools
for materials informatics. Its functionalities partially depend on Python (PyTorch) and R (MXNet).
This package is still under development. The current released version provides some limited features:

* Interface to the public materials database
* Library of materials descriptors (compositional/structural descriptors)
* pre-trained model library **XenonPy.MDL** (v0.1.0.beta, 2019/8/9: more than 140,000 models (include private models) in 35 properties of small molecules, polymers, and inorganic compounds) [Currently under major maintenance, expected to be recovered in v0.7]
* Machine learning tools.
* Transfer learning using the pre-trained models in XenonPy.MDL


.. image:: _static/xenonpy.png


--------
Citation
--------
XenonPy is an on-going research project that covers multiple important topics in materials informatics.
We recommend users to cite the papers that are relevant to their specific use of XenonPy.
Please refer to :doc:`features` for details of each feature in XenonPy with its corresponding citation.
User can also check the publication list below to pick the relevant citations.


--------
Features
--------
XenonPy has a rich set of tools for various materials informatics applications.
The descriptor generator class can calculate several types of numeric descriptors from ``compositional``, ``structure``.
By using XenonPy's built-in visualization functions, the relationships between descriptors and target properties can be easily shown in a heatmap.
XenonPy also supports an interface to use the ``rdkit`` descriptors and provides the ``iQSPR`` algorithm for molecular design.

Transfer learning is an important tool for the efficient application of machine learning methods to materials informatics.
To facilitate the widespread use of transfer learning,
we have developed a comprehensive library of pre-trained models, called **XenonPy.MDL**.
This library provides a simple API that allows users to fetch the models via an HTTP request.
For the ease of using the pre-trained models, some useful functions are also provided.

See :doc:`features` for details.


------
Sample
------
Sample codes of different features in XenonPy are available here: https://github.com/yoshida-lab/XenonPy/tree/master/samples


.. _user-doc:
.. toctree::
    :maxdepth: 2
    :caption: User Documentation

    books
    copyright
    installation
    features
    tutorial
    api
    contribution
    changes
    contact


------------
Publications
------------

.. [1] H. Ikebata, K. Hongo, T. Isomura, R. Maezono, and R. Yoshida, “Bayesian molecular design with a chemical language model,” J Comput Aided Mol Des, vol. 31, no. 4, pp. 379–391, Apr. 2017, doi: 10/ggpx8b.
.. [2] S. Wu et al., “Machine-learning-assisted discovery of polymers with high thermal conductivity using a molecular design algorithm,” npj Computational Materials, vol. 5, no. 1, pp. 66–66, Dec. 2019, doi: 10.1038/s41524-019-0203-2.
.. [3] S. Wu, G. Lambard, C. Liu, H. Yamada, and R. Yoshida, “iQSPR in XenonPy: A Bayesian Molecular Design Algorithm,” Mol. Inform., vol. 39, no. 1–2, p. 1900107, Jan. 2020, doi: 10.1002/minf.201900107.
.. [4] H. Yamada et al., “Predicting Materials Properties with Little Data Using Shotgun Transfer Learning,” ACS Cent. Sci., vol. 5, no. 10, pp. 1717–1730, Oct. 2019, doi: 10.1021/acscentsci.9b00804.
.. [5] S. Ju et al., “Exploring diamondlike lattice thermal conductivity crystals via feature-based transfer learning,” Phys. Rev. Mater., vol. 5, no. 5, p. 053801, May 2021, doi: 10.1103/physrevmaterials.5.053801.
.. [6] C. Liu et al., “Machine Learning to Predict Quasicrystals from Chemical Compositions,” Adv. Mater., vol. 33, no. 36, p. 2102507, Sep. 2021, doi: 10.1002/adma.202102507.


------------
Contributing
------------

XenonPy is an `open source project <https://github.com/yoshida-lab/XenonPy>`_ inspired by `matminer <https://hackingmaterials.github.io/matminer>`_.
:raw-html:`<br/>`
This project is under continuous development. We would appreciate any feedback from the users.
:raw-html:`<br/>`
Code contributions are also very welcomed. See :doc:`contribution` for more details.


.. _マテリアルズインフォマティクス: https://www.kyoritsu-pub.co.jp/book/b10013510.html
.. _KYORITSU SHUPPAN: https://www.kyoritsu-pub.co.jp/
.. _pandas: https://pandas.pydata.org
.. _PyTorch: http://pytorch.org/
.. _Xenon: https://en.wikipedia.org/wiki/Xenon

.. |MacOS| image:: https://github.com/yoshida-lab/XenonPy/workflows/MacOS/badge.svg
    :alt: MacOS Building Status
    :target: https://github.com/yoshida-lab/XenonPy/actions?query=workflow%3AMacOS

.. |Windows| image:: https://github.com/yoshida-lab/XenonPy/workflows/Windows/badge.svg
    :alt: Windows Building Status
    :target: https://github.com/yoshida-lab/XenonPy/actions?query=workflow%3AWindows

.. |Ubuntu| image:: https://github.com/yoshida-lab/XenonPy/workflows/Ubuntu/badge.svg
    :alt: Ubuntu Building Status
    :target: https://github.com/yoshida-lab/XenonPy/actions?query=workflow%3AUbuntu

.. |readthedocs| image:: https://readthedocs.org/projects/xenonpy/badge/?version=latest
    :alt: Documentation Status
    :target: https://xenonpy.readthedocs.io/en/latest/?badge=latest

.. |codecov| image:: https://codecov.io/gh/yoshida-lab/XenonPy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/yoshida-lab/XenonPy

.. |version| image:: https://img.shields.io/github/tag/yoshida-lab/XenonPy.svg?maxAge=360
    :alt: Version
    :target: https://github.com/yoshida-lab/XenonPy/releases/latest

.. |python| image:: https://img.shields.io/pypi/pyversions/xenonpy.svg
    :alt: Python Versions
    :target: https://pypi.org/project/xenonpy/

.. |total-dl| image:: https://pepy.tech/badge/xenonpy
    :alt: Downloads
    :target: https://pepy.tech/badge/xenonpy

.. |per-dl| image:: https://img.shields.io/pypi/dm/xenonpy.svg?label=PiPy%20downloads
    :alt: PyPI - Downloads

