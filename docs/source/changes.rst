.. role:: raw-html(raw)
    :format: html

=======
Changes
=======

v0.5.0
======

**Breaking change**

* Remove the ``BoxCox`` transform. ( `#222`_ )

**New features**

* Add new contribution ``iSMD`` by `qi`_. ( `#208`_ )
* Add classification support for ``Trainer``. ( `#184`_ )

**Enhance**

* ``BaseFeaturizer.transform`` now supports ``pandas.DataFrame`` as input. (??)
* Allow using custom elemental information matrix in ``Compositions`` descriptor. ( `#221`_ )
* Use ``joblib.parallel`` as default parallel backend. ( `#191`_, `#220`_ )
* ``Splliter.split`` method now support python list as input. ( `#194`_ )
* Allow user specific index for ``DescriptorHeatmap``. ( `#44`_ )
* Allow control of number of layers to be extracted in ``FrozenFeaturizer``. ( `#147`_ )


**Infrastructure improve**

* Move CI to github action. ( `#195`_ )
* Move to readthedocs version 2. ( `#206`_ )

.. _qi: https://github.com/qi-zh
.. _#222: https://github.com/yoshida-lab/XenonPy/pull/222
.. _#208: https://github.com/yoshida-lab/XenonPy/pull/208
.. _#221: https://github.com/yoshida-lab/XenonPy/pull/221
.. _#184: https://github.com/yoshida-lab/XenonPy/pull/184
.. _#195: https://github.com/yoshida-lab/XenonPy/pull/195
.. _#206: https://github.com/yoshida-lab/XenonPy/pull/206
.. _#191: https://github.com/yoshida-lab/XenonPy/pull/191
.. _#220: https://github.com/yoshida-lab/XenonPy/pull/220
.. _#194: https://github.com/yoshida-lab/XenonPy/pull/194
.. _#44: https://github.com/yoshida-lab/XenonPy/pull/44
.. _#174: https://github.com/yoshida-lab/XenonPy/pull/174


v0.4.2
======

**Bug fix**

* fix ``set_param`` method dose not set the children in ``BaseDescriptor`` and ``NGram``. ( `#163`_, `#159`_ )

**Enhance**

* Setting ``optimizer``, ``loss_func``, and etc. can be done in ``Trainer.load``. ( `#158`_ )
* Improve docs.  ( `#155`_ )

.. _#163: https://github.com/yoshida-lab/XenonPy/issues/163
.. _#159: https://github.com/yoshida-lab/XenonPy/issues/159
.. _#158: https://github.com/yoshida-lab/XenonPy/issues/159
.. _#155: https://github.com/yoshida-lab/XenonPy/issues/159


v0.4.0
======

**Breaking change**

* Remove ``xenonpy.datatools.MDL``.
* Remove ``xenonpy.model.nn`` modules. Part of them will be kept until v1.0.0 for compatible.

**New features**

* Add ``xenonpy.mdl`` modules for XenonPy.MDL access.
* Add ``xenonpy.model.training`` modules for model training.


v0.3.6
======

**Breaking change**

* Renamed ``BayesianRidgeEstimator`` to ``GaussianLogLikelihood``.
* Removed the ``estimators`` property from ``BayesianRidgeEstimator``.
* Added ``predict`` method into ``GaussianLogLikelihood``.


v0.3.5
======

**Enhanced**

* Added version specifiers to the *requirements.txt* file.

v0.3.4
======

**Bug fix**

* Fixed a critical error in ``BayesianRidgeEstimator`` when calculating the loglikelihood. ( `#124`_ )

.. _#124: https://github.com/yoshida-lab/XenonPy/issues/124

v0.3.3
======

**Bug fix**

* fix *mp_ids.txt* not exist error when trying to build the sample data using ``preset.build``.

v0.3.2
======

**Enhanced**

* Updated sample codes.
* Added progress bar for ngram training. ( `#93`_ )
* Added error handling to NGram when generating new SMILES. ( `#97`_ )

**CI**

* Removed python 3.5 support. ( `#95`_ )
* Added Appveyor CI for windows tests. ( `#90`_ )

.. _#93: https://github.com/yoshida-lab/XenonPy/issues/93
.. _#97: https://github.com/yoshida-lab/XenonPy/issues/97
.. _#95: https://github.com/yoshida-lab/XenonPy/issues/95
.. _#90: https://github.com/yoshida-lab/XenonPy/issues/90


v0.3.1
======

**Enhanced**

* Added tutorials for main modules. ( `#79`_ )

.. _#79: https://github.com/yoshida-lab/XenonPy/issues/79


v0.3.0
======

**Breaking changes**:

* Removed Built-in data ``mp_inorganic``, ``mp_structure``, ``oqmd_inorganic`` and ``oqmd_structure``. ( `#12`_, `#20`_ )
* Renamed ``LocalStorage`` to ``Storage``.

**Enhanced**

* Added error handling for ``NGram`` training. ( `#75`_, `#86`_ )
* Added error handling for ``IQSPR``. ( `#69`_ )
* Added error handling for ``BaseDescriptor`` and ``BaseFeaturizer``. ( `#73`_ )
* Added featurizer selection function. ( `#47`_ )

**New Features**

* Added sample data building function for ``preset``. ( `#81`_, `#84`_ )


.. _#12: https://github.com/yoshida-lab/XenonPy/issues/12
.. _#20: https://github.com/yoshida-lab/XenonPy/issues/20
.. _#75: https://github.com/yoshida-lab/XenonPy/issues/75
.. _#73: https://github.com/yoshida-lab/XenonPy/issues/73
.. _#86: https://github.com/yoshida-lab/XenonPy/issues/86
.. _#69: https://github.com/yoshida-lab/XenonPy/issues/69
.. _#81: https://github.com/yoshida-lab/XenonPy/issues/81
.. _#84: https://github.com/yoshida-lab/XenonPy/issues/84
.. _#47: https://github.com/yoshida-lab/XenonPy/issues/47




v0.2.0
======

**Descriptor Generator**:

* Added ``xenonpy.descriptor.Fingerprint`` descriptor generator. ( `#21`_ )
* Added ``xenonpy.descriptor.OrbitalFieldMatrix`` descriptor generator. ( `#22`_ )


**API Changes**:

* Allowed ``BaseDescriptor`` class to use anonymous/renamed input. ( `#10`_ )

.. _#10: https://github.com/yoshida-lab/XenonPy/issues/10
.. _#21: https://github.com/yoshida-lab/XenonPy/issues/21
.. _#22: https://github.com/yoshida-lab/XenonPy/issues/22