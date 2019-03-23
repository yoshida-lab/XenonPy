======================
Descriptor calculation
======================

**XenonPy** comes with a general interface for descriptor calculation.
By using this interface, users can implement their descriptor calculator with only a few lines of codes and run it smoothly.

We also use this system to provide built-in calculators. Currently, 15 featurizers in 4 types are available out-of-the-box.
The following list shows a summary.

.. csv-table:: Summary of built-in featurizers
    :header: "Featurizer", "Type", "Description"

    "WeightedAvgFeature", "Composition", "Weighted average (abbr: ave): :math:`f_{ave, i} = w_{A}^* f_{A,i} + w_{B}^* f_{B,i}`"
    "WeightedSumFeature", "Composition", "Weighted variance (abbr: var): :math:`f_{var, i} = w_{A}^* (f_{A,i} - f_{ave, i})^2  + w_{B}^* (f_{B,i} - f_{ave, i})^2`"
    "WeightedVarFeature", "Composition", "Max-pooling (abbr: max): :math:`f_{max, i} = max{f_{A,i}, f_{B,i}}`"
    "MaxFeature", "Composition", "Min-pooling (abbr: min): :math:`f_{min, i} = min{f_{A,i}, f_{B,i}}`"
    "MinFeature", "Composition", "Weighted sum (abbr: sum): :math:`f_{sum, i} = w_{A} f_{A,i} + w_{B} f_{B,i}`"
    "RDKitFP", "Fingerprint", "RDKit fingerprint"
    "AtomPairFP", "Fingerprint", "Atom Pair fingerprints"
    "MACCS", "Fingerprint", "The MACCS keys for a molecule"
    "ECFP", "Fingerprint", "Morgan (Circular) fingerprints (ECFP)"
    "FCFP", "Fingerprint", "Morgan (Circular) fingerprints + feature-based (FCFP)"
    "TopologicalTorsionFP", "Fingerprint", "Topological Torsion fingerprints"
    "ObitalFieldMatrix", "Structure", "Representation based on the valence shell electrons of neighboring atoms"
    "RadialDistributionFunction", "Structure", "Radial distribution in crystal"
    "FrozenFeaturizer", "NN Feature", "Neural Network Extracted Feature"


-------------------------
Compositional descriptors
-------------------------

XenonPy can calculate 290 compositional features for a given chemical composition.
This calculation uses information of the 58 element-level property data recorded in ``elements_completed``.
See :ref:`features:Data access` for details.

    >>> from xenonpy.descriptor import Compositions
    >>> cal = Compositions()
    >>> cal
    Compositions:
      |- composition:
      |  |- WeightedAvgFeature
      |  |- WeightedSumFeature
      |  |- WeightedVarFeature
      |  |- MaxFeature
      |  |- MinFeature

The structure information of calculator ``Cal`` is shown above.
This information tells us ``Cal`` has one featurizer group called **composition** with featurizers
``WeightedAvgFeature``, ``WeightedSumFeature``, ``WeightedVarFeature``, ``MaxFeature`` and ``MinFeature`` in it.

To use this calculator, users have to structure an iterable object that contains the information of compounds' composition, then feed it to the method ``transform`` or ``fit_transform`` in ``cal``.
These methods accept two types of input, the ``pymatgen.Structure`` objects, or dicts which have the structure like `{'H': 2, 'O': 1}`.

Using our sample data, users will obtain a pandas.DataFrame object that contains all the compositional descriptors.

    >>> from xenonpy.datatools import preset
    >>> samples = preset.mp_samples
    >>> comps = samples['composition']
    >>> descriptor = cal.transform(comps)
    >>> descriptor
         ave:atomic_number  ...  min:Polarizability
    0            24.666667  ...            0.802000
    1            33.000000  ...            1.100000
    2            21.600000  ...            0.802000
    ...                ...  ...                 ...
    928          44.500000  ...            5.500000
    929          24.250000  ...           25.000000
    930          26.750000  ...            4.800000
    931          36.000000  ...            6.600000
    932          16.500000  ...            0.802000
    [933 rows x 290 columns]

where

    >>> comps.__class__
    pandas.core.series.Series
    >>> comps[0].__class__
    dict


If the input is a pandas.DataFrame object, the calculator will first try to read the data columns that have the same name as the featurizer groups. For example, the name of the featurizer group in the example above is **composition**. Therefore, the whole object entry can be fed into the calculator's methods without explicitly extracting the **composition** column in the ``samples``:

    >>> descriptor = cal.transform(samples)
    >>> descriptor
         ave:atomic_number  ...  min:Polarizability
    0            24.666667  ...            0.802000
    1            33.000000  ...            1.100000
    2            21.600000  ...            0.802000
    ...                ...  ...                 ...
    928          44.500000  ...            5.500000
    929          24.250000  ...           25.000000
    930          26.750000  ...            4.800000
    931          36.000000  ...            6.600000
    932          16.500000  ...            0.802000
    [933 rows x 290 columns]

This do the same work as the previous one.


----------------------
Structural descriptors
----------------------

Similar to the ``Compositions`` calculator, ``Structures`` accepts ``pymatgen.Structure`` objects as its input, and then return calculated results as a pandas.DataFrame.

    >>> from xenonpy.descriptor import Structures
    >>> cal = Structures()
    >>> cal
    Structures:
      |- structure:
      |  |- RadialDistributionFunction
      |  |- ObitalFieldMatrix

``Structures`` contains one featurizer group called **structure** with ``RadialDistributionFunction`` and ``ObitalFieldMatrix`` in it.
``samples`` also has the structure information. We can use these to calculate structural descriptors.

    >>> descriptor = cal.transform(samples)

This will take 3 ~ 5 min to run and finally you will get:

    >>> descriptor.head(5)
                0.1  0.2  0.30000000000000004  ...  f14_f12  f14_f13  f14_f14
    mp-1008807  0.0  0.0                  0.0  ...      0.0      0.0   0.0000
    mp-1009640  0.0  0.0                  0.0  ...      0.0      0.0   0.0000
    mp-1016825  0.0  0.0                  0.0  ...      0.0      0.0   0.0000
    mp-1017582  0.0  0.0                  0.0  ...      0.0      0.0   0.3851
    mp-1021511  0.0  0.0                  0.0  ...      0.0      0.0   0.0000
    [5 rows x 1224 columns]


-------
Advance
-------

There are more details of the descriptor calculator system that are not yet included in this tutorial.
Before we complete this document, you can check out https://github.com/yoshida-lab/XenonPy/blob/master/samples/build_your_own_descriptor_calculator.ipynb for more information.
