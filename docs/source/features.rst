.. role:: raw-html(raw)
    :format: html

========
Features
========


Element-level features
======================

XenonPy uses two sets of element-level features, elements and elements_completed (imputed),
to calculate a descriptor vector for a given chemical composition of an input material.
The data were collected from mendeleev, pymatgen, CRC Hand Book and Magpie.

elements contains 74 element-level properties of 118 elements in which their missing values
were statistically imputed by performing the multiple imputation method [1]_, giving elements_completed.
Because of the statistical unreliability in the missing impution for a subset of properties and heavier atoms that contained many missing values in elements,
the elements_completed data set provides only 58 features of the 94 elements from H to Pu.

To get the data sets, see quick examples at:
:raw-html:`<br />`
https://github.com/yoshida-lab/XenonPy/blob/master/samples/load_and_save_data.ipynb

The :doc:`dataset` shows the currently available elemental information.

**Reference**

.. [1] Rubin DB. Multiple imputation for nonresponse in surveys. New York: Wiley; 1987.



Compositional descriptors
=========================

XenonPy calculates 290 compositional features for a given chemical composition.
This calculation use the additional information of 58 element-level features recorded in the built-in elements_completed.
For example, let’s consider a binary compound, :math:`A_{w_A}B_{w_B}`, whose the element-level features are denoted by :math:`f_{A,i}` and :math:`f_{B,i} (i = 1, …, 58)`. Then, the 290 compositional descriptors are calculated: for :math:`i = 1, …, 58`,

* Weighted average (abbr: ave): :math:`f_{ave, i} = w_{A}^* f_{A,i} + w_{B}^* f_{B,i}`,
* Weighted variance (abbr: var): :math:`f_{var, i} = w_{A}^* (f_{A,i} - f_{ave, i})^2  + w_{B}^* (f_{B,i} - f_{ave, i})^2`,
* Max-pooling (abbr: max): :math:`f_{max, i} = max{f_{A,i}, f_{B,i}}`, 
* Min-pooling (abbr: min): :math:`f_{min, i} = min{f_{A,i}, f_{B,i}}`,
* Weighted sum (abbr: sum): :math:`f_{sum, i} = w_{A} f_{A,i} + w_{B} f_{B,i}`,

where :math:`w_{A}^*` and :math:`w_{B}^*` denote the normalized composition summing up to one.

See the sample code for calculating the compositional descriptors, illustrated on 69,640 compounds in Materials Project:
:raw-html:`<br />`
https://github.com/yoshida-lab/XenonPy/blob/master/samples/load_and_save_data.ipynb



Structural descriptors
======================

Currently, XenonPy implements RDF (radial distribution function) and OFM (orbital field matrix [2]_) as the descriptors of crystalline structures.
We also provide compatible API to use the structural descriptors of `matminer <https://hackingmaterials.github.io/matminer/>`_.
Follow this link to check the `Table of Featurizers <https://hackingmaterials.github.io/matminer/featurizer_summary.html>`_ in matminer.

See the sample code for calculating the structural descriptors, illustrated on 69,640 compounds in Materials Project:
:raw-html:`<br />`
https://github.com/yoshida-lab/XenonPy/blob/master/samples/load_and_save_data.ipynb


**Reference**

.. [2] Pham et al. Machine learning reveals orbital interaction in materials, Sci Technol Adv Mater. 18(1): 756-765, 2017.




Visualization of descriptor-property relationships
==================================================

Descriptors on a set of given materials could be displayed on a heatmap plot in order to facilitate the understanding of overall patterns in relation to their properties.
The following figure shows an example:

.. figure:: _static/heatmap.jpg

     Heatmap of 290 compositional descriptors of 69,640 compounds in Materials Project(upper: volume Å\ :sup:`3`\ , lower:  density g/cm\ :sup:`3`\  ).

In the heatmap display of the descriptor matrix, the 69,640 materials are arranged from the top to bottom by the increasing order of formation energies.
Plotting the descriptor-property relationships in this way, we could visually recognize which descriptors are relevant or irrelevant to the prediction of formation energies.
Relevant descriptors, which are linearly or nonlinearly dependent to formation energies, might exhibit certain patterns from top to bottom in the heatmap.
For example, a monotonically decreasing or increasing pattern would appear in a linearly dependent descriptor.
On the other hand, irrelevant descriptors might exhibit no specific patterns.

See the sample code on the visualization of descriptor-property relationships:
:raw-html:`<br />`
https://github.com/yoshida-lab/XenonPy/blob/master/samples/load_and_save_data.ipynb



XenonPy.MDL and Transfer Learning
=================================

XenonPy.MDL is a library of pre-trained models that were obtained by feeding diverse materials data on structure-property relationships into neural networks and some other supervised learning algorithms.
The current release (version 0.1.0) contains more than 100,000 models on physical, chemical, electronic, thermodynamic, or mechanical properties of
small organic molecules (15 properties), polymers/polymer composites (18), and inorganic compounds (12).
Pre-trained neural networks are distributed as either the R (MXNet) or Python model objects (PyTorch).
For details (a list of models, properties, source data used for training, and so on), see the paper [3]_.

Transfer learning is an increasingly popular framework of machine learning that covers a broad range of methodologies for which a model trained on one task is re-purposed on another related task [4]_ [5]_. In general, the need for transfer learning occurs when there is a limited supply of training data, but there are many other promising applications in materials science as described in [1]. Given a target property 
By using the transfer learning module of XenonPy, the models can be used as the machine learning acquired descriptors (the neural decscriptors) as demonstrated in [3]_.

The usage information will be released later.


**Reference**

.. [3] Yamada, H., Liu, C., Wu, S., Koyama, Y., Ju, S., Shiomi, J., Morikawa, J., Yoshida, R. Transfer learning: a key driver of accelerating materials discovery with machine learning, in preparation.
.. [4] Karl, W.; Khoshgoftaar, T. M.; Wang, D. J. of Big Data 2016, 3, 1–40.
.. [5] Chuanqi, T.; Fuchun, S.; Tao, K.; Wenchang, Z.; Chao, Y.; Chunfang, L. arXiv 2018, abs/1808.01974 .
