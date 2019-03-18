=============
Visualization
=============

Descriptors on a set of given materials could be displayed as a heatmap.
By using this plotting, the descriptor-property relationships can be understand.
This tutorial will show how to draw a heatmap.


------------------
Descriptor heatmap
------------------

We will use ``Composition`` descriptors and property ``density`` to demonstrate heatmap drawing step-by-step.

1. calculate descriptors

    .. code-block:: python

        from xenonpy.datatools import preset
        from xenonpy.descriptor import Compositions

        samples = preset.mp_samples
        cal = Compositions()

        descriptor = cal.transform(samples['composition'])

2. sort descriptors by property values

    .. code-block:: python

        # use formation energy as our target
        property_ = 'density'

        # --- sort property
        prop = samples[property_].dropna()
        sorted_prop = prop.sort_values()

        # --- sort descriptors
        sorted_desc = descriptor.loc[sorted_prop.index]

3. draw the heatmap

    .. code-block:: python

        # --- import necessary libraries

        from xenonpy.visualization import DescriptorHeatmap

        heatmap = DescriptorHeatmap(
                bc=True,  # use box-cox transform
                save=dict(fname='heatmap_density', dpi=200, bbox_inches='tight'),  # save fingure to file
                figsize=(70, 10))
        heatmap.fit(sorted_desc)
        heatmap.draw(sorted_prop)

Finally, we got the picture below.

.. figure:: ../_static/heatmap_density.png

     Heatmap of 290 compositional descriptors of 933 compounds in Materials Project (density g/cm\ :sup:`3`\  ).

In the heatmap of the descriptor matrix, the materials are arranged from the top to bottom by the increasing order
of density. Plotting the descriptor-property relationships in this way, we could visually recognize which
descriptors are relevant or irrelevant to the prediction of formation energies. Relevant descriptors, which are linearly
or nonlinearly dependent to formation energies, might exhibit certain patterns from top to bottom in the heatmap. For example,
a monotonically decrease or increase pattern would appear in a linearly dependent descriptor. On the other hand,
irrelevant descriptors might exhibit no specific patterns.

See run able sample at:

    https://github.com/yoshida-lab/XenonPy/blob/master/samples/visualization.ipynb