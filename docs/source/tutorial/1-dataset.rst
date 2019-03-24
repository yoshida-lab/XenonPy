===========
Data access
===========

**Dataset** is an abstraction of local file system.
Users can add their local paths into this system to easily access the data inside.
The basic concept is to treat a data file as a property of a ``Dataset`` object.
The following docs show how easy it is to interact with the data in system.


-------
Dataset
-------

Assuming that you have some data files ``data1.csv``, ``data1.pkl.pd_``, ``data1.pkl.z_`` under dir `/set1`
and ``data2.csv``, ``data2.pkl.pd_``, ``data2.pkl.z_`` under dir `/set2`.

The following codes will create a ``Dataset`` object containing all available files under `/set1` and `/set2`.

    >>> from xenonpy.datatools import Dataset
    >>> dataset = Dataset('/set1', '/set2')
    >>> dataset
    <Dataset> includes:
    "data1": /set1/data1.pkl.pd_
    "data2": /set2/data2.pkl.pd_

Now, you can retrieve data by their name like this:

    >>> dataset.dataframe.data1

What the code did is that, the ``dataset`` loaded a file with name ``data1.pkl.pd_`` from `/set1` or `/set2`.
In this case, the `/set1/data1.pkl.pd_` was loaded.

It is important to note that we called a property named ``dataframe`` before we load ``data1`` in order to let ``dataset`` know that it is loading a ``pandas.DataFrame`` object file using the ``pd.read_pickle`` function.

Currently, 4 loaders are available out-of-the-box. The information of built-in loaders are summarised as below.

.. table:: built-in loaders

    ==============  ==================  =============================
    file extension        loader              description
    --------------  ------------------  -----------------------------
    ``pkl.pd_``     pd.read_pickle      pandas.DataFrame object file
    ``csv``         pd.read_csv         csv file
    ``xlsx``        pd.read_excel       excel file
    ``pkl.z_``      joblib.load         common pickled files
    ==============  ==================  =============================

The default loader is ``dataframe``. This means that if you want to load a pandas.DataFrame object, you can omit the ``dataframe``.
The following code exactly do the same work as explained above:

    >>> dataset.data1

You can also specify the default loader by setting the ``backend`` parameter:

    >>> dataset = Dataset('set1', 'set2', backend='csv')
    >>> dataset.data1  # this will load '/set1/data1.csv'



------
preset
------

XenonPy also uses this system to provide some built-in data.
Currently, two sets of element-level property data are available out-of-the-box (``elements`` and ``elements_completed`` (imputed version of ``elements``)).
These data were collected from `mendeleev`_, `pymatgen`_, `CRC Hand Book`_ and `Magpie`_.
To know the details of ``elements_completed``, see :ref:`features:Data access`

.. _CRC Hand Book: http://hbcponline.com/faces/contents/ContentsSearch.xhtml
.. _Magpie: https://bitbucket.org/wolverton/magpie
.. _mendeleev: https://mendeleev.readthedocs.io
.. _pymatgen: http://pymatgen.org/

Use the following codes to load ``elements`` and ``elements_completed``.

    >>> from xenonpy.datatools import preset
    >>> preset.elements
    >>> preset.elements_completed

These are still some advance uses of ``Dataset`` and ``preset``. For more details, see :ref:`tutorial/1-dataset:Advance`.

Also see the jupyter files at:

    https://github.com/yoshida-lab/XenonPy/tree/master/samples/dataset_and_preset.ipynb


-------
Storage
-------

For implementation details, you can check out our sample codes:

    https://github.com/yoshida-lab/XenonPy/tree/master/samples/storage.ipynb




-------
Advance
-------

Coming soon!
