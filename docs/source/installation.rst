============
Installation
============
XenonPy can be used in Python 3.5, 3.6 and 3.7.
We have no plane to support Python 2.x (also `pymatgen library will discontinuing Python 2 support from 2018 <http://pymatgen.org/#py3k-only-with-effect-from-2019-1-1>`_).
For some legacy clusters which only come with Python 2.x, a possibility approach is to create an isolated Python 3.5 or 3.6 environment.

XenonPy use PyTorch_ as the NN engine backend because of it's flexibility and high performance.
Since PyTorch has no common installation recipe beacuse it dependent on what your OS is and either use `CUDA <https://developer.nvidia.com/cuda-toolkit>`_.
So unfortunately, you have to install PyTorch manually after the installation of XenonPy.


Let we say:

* :ref:`install_xenonpy` 
* :ref:`install_pytorch`

.. note::

    When installation, some dependence which have C/Cython files need to be compiled.
    On Unix-like system is actually quite simple only need you to install compile toolchain. For windows the installation is relative triviality.
    Before you install python packages. First install `Visual C++ Build Tools <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_.
    Also you have to checked on **windows 8.1 / 10 SDK options** when install Visual C++ Build Tools.


.. _install_xenonpy:

Step1: Install XenonPy
=======================


via pip
--------------------------

A common way to install python packages is pip_.

Simply run the command in a bash terminal:

.. code-block:: bash

    $ pip install xenonpy

or, to install XenonPy in your user home folder, run the command:

.. code-block:: bash

    $ pip install xenonpy --user

To update, simply type

.. code-block:: bash

    $ pip install --upgrade xenonpy




Install in development mode
-----------------------------

To install from the latest source code of the XenonPy from our `Github repository`_, clone the Git source:

.. code-block:: bash

    $ git clone https://github.com/yoshida-lab/XenonPy.git

and then enter the cloned repository/folder to install in developer mode:

.. code-block:: bash

    $ cd XenonPy
    $ pip install -e .

To update XenonPy, enter your cloned folder and type

.. code-block:: bash

    $ git pull
    $ pip install -e .



.. _install_pytorch:

Step2: Install PyTorch
======================
.. image:: _static/pytorch.png
    :alt: PyTorch installation
    :target: http://pytorch.org/
    :width: 800

As mentioned above you have to install PyTorch manually. This picture show various combinations of how to install PyTorch.
Please follow the `Get Started` section on the PyTorch_ official sit.



Troubleshooting/Issues
======================

Having issues installing? Contract us at issues_ and Gitter_.

When open issues, please describing your problem in full
(including your system specifications, Python version information, and input/output log).

-----------------------------------------------------------------------------------------------------------

.. _conda: http://conda.pydata.org
.. _yoshida-lab channel: https://anaconda.org/yoshida
.. _pip: https://pip.pypa.io
.. _Github repository: https://github.com/yoshida-lab/XenonPy
.. _issues: https://github.com/yoshida-lab/XenonPy/issues
.. _Gitter: https://gitter.im/yoshida-lab/XenonPy
.. _PyTorch: http://pytorch.org/
