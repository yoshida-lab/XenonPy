========
Tutorial
========

These tutorials will demonstrate the most foundational usages of XenonPy and XenonPy.MDL and
we assume that you can use `Jupyter notebook <https://jupyter.org/>`_.

.. toctree::
   :maxdepth: 1
   :glob:

   tutorial/*


-----------
Sample data
-----------

Before starting these tutorials, you need to prepare some sample data for the benchmark/test.

We selected 1,000 inorganic compounds randomly from the `Materials Project <https://materialsproject.org>`_ database for this using.
You can check these **MP ids** at:

    https://github.com/yoshida-lab/XenonPy/blob/master/samples/mp_ids.txt.

To build the sample data, you also have to create an API key at the materials project.
Please see `The Materials API <https://materialsproject.org/open>`_  and follow the official documents.


You can use the following codes to build your sample data. These data will be saved at ``~/.xenonpy/userdata``.

    >>> from xenonpy.datatools import preset
    >>> preset.build('mp_samples', api_key='your api key')

If you want to know what exactly the code did, please check:

    https://github.com/yoshida-lab/XenonPy/blob/master/samples/build_sample_data.ipynb

