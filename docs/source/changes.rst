.. role:: raw-html(raw)
    :format: html

=======
Changes
=======

v0.6.1
======

**Breaking change**

* ``GaussianNLLLoss`` has been removed from ``xenonpy.model.training.loss``. ( `#249`_ )

.. _#249: https://github.com/yoshida-lab/XenonPy/pull/249

v0.6.0
======

**Bug fix**

* Fix a bug in the ``Scaler`` class. ( `#243`_ )

**New features**

* Add ``average`` option to the ``classification_metrics`` function. Now users can decide how to calculate scores for multilabel tasks. ( `#240`_ )
* Add ``only_best_states`` option to the ``Persist`` class. If ``True``, ``Persist`` will only save the best state to reduce the storage space. ( `#233`_ )
* Add ``warming_up`` option to the ``Validator`` class. ( `#238`_ )

.. _#243: https://github.com/yoshida-lab/XenonPy/pull/243
.. _#240: https://github.com/yoshida-lab/XenonPy/pull/240
.. _#233: https://github.com/yoshida-lab/XenonPy/pull/233
.. _#238: https://github.com/yoshida-lab/XenonPy/pull/238

v0.5.2
======

**Bug fix**

* some minor fixes.

v0.5.1
======

**Bug fix**

* update the pip install dependencies in ``requirements.txt``. ( `#226`_ )

**Enhance**

* Add ``rdkit v2020.09`` support for fingerprint descriptors. ( `#228`_ )
* Add return probability support for ``model.extension.TensorConverter``. ( `#229`_ )

.. _#226: https://github.com/yoshida-lab/XenonPy/pull/226
.. _#228: https://github.com/yoshida-lab/XenonPy/pull/228
.. _#229: https://github.com/yoshida-lab/XenonPy/pull/229

v0.5.0
======

**Breaking change**

* Replace ``xenonpy.datatools.BoxCox`` with ``xenonpy.datatools.PowerTransformer``. ( `#222`_ )

**New features**

* Add ``xenonpy.datatools.PowerTransformer`` to provide *yeo-johnson* and *box-cox* transformation through the ``sklearn.preprocessing.PowerTransformer``. ( `#222`_ )
* Add new contribution ``ISMD`` by `Qi`_, a new class of ``BaseProposal`` that allows generation of molecules based on virtual reaction of reactants in a predefined reactant pool. ( `#208`_ )
* Add classifier training support for the ``xenonpy.model.Trainer``. ( `#184`_ )
* Add ``IQSPR4DF`` to support ``pandas.DataFrame`` input to iQSPR.
* Add ``LayeredFP``, ``PatternFP``, and ``MHFP`` (new rdkit fingerprints).

**Enhance**

* ``BaseFeaturizer.transform`` now supports ``pandas.DataFrame`` as input where relevant columns for descriptor calculation can be specified through ``target_col``.
* ``BaseDescriptor.transform`` and ``BaseLogLikelihoodSet.log_likelihood`` now automatically check if any group names occur in the input ``pandas.DataFrame`` column names. If not, the entire ``pandas.DataFrame`` will be passed to the corresponding ``BaseFeaturizer`` and ``BaseLogLikelihood``, respectively.
* Allow using custom elemental information matrix in ``xenonpy.descriptor.Compositions`` descriptor. ( `#221`_ )
* Use ``joblib.parallel`` as default parallel backend. ( `#191`_, `#220`_ )
* ``Splliter.split`` method now support python list as input. ( `#194`_ )
* Allow user specific index for ``DescriptorHeatmap``. ( `#44`_ )
* Allow control of number of layers to be extracted in ``FrozenFeaturizer``. ( `#174`_ )
* ``bit_per_entry`` option is added to ``RDKitFP`` and ``AtomPairFP`` to allow control of number of bits to represent one fingerprint entry.
* ``counting`` option is added to ``RDKitFP``, ``AtomPairFP``, ``TopologicalTorsionFP``, ``FCFP`` and ``ECFP`` to support returning counts of each fingerprint entry.
* Column names of ``DescriptorFeature`` is updated to be consistent with the rdkit naming.


**Infrastructure improve**

* Move CI to github action. ( `#195`_ )
* Move to readthedocs version 2. ( `#206`_ )

.. _Qi: https://github.com/qi-zh
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
