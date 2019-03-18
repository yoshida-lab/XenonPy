.. role:: raw-html(raw)
    :format: html

=======
Changes
=======


v0.3.0
======

**Breaking changes**:

* Remove Built-in data ``mp_inorganic``, ``mp_structure``, ``oqmd_inorganic`` and ``oqmd_structure``. ( `#12`_ `#20`_ )
* Change name of ``LocalStorage`` to ``Storage``.

**Enhanced**

* Add error handling for ``NGram`` training. ( `#75`_ `#86`_ )
* Add error handling for ``IQSPR``. ( `#69`_ )
* Add error handling for ``BaseDescriptor`` and ``BaseFeaturizer``. ( `#73`_ )
* Add featurizer selection function. ( `#47`_ )

**New Features**

* Add sample data building function for ``preset``. ( `#81`_ `#84`_ )


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