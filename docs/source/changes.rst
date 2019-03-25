.. role:: raw-html(raw)
    :format: html

=======
Changes
=======

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