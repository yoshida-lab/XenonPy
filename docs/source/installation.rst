.. role:: raw-html(raw)
    :format: html

============
Installation
============

XenonPy can be installed using pip_ in Python 3.5, 3.6 and 3.7 on Mac, Linux and Windows.
Alternatively, we recommend using the `Docker Image`_ if you have no installation preference.

We have no plan to support Python 2.x. One of the main reasons is that the ``pymatgen`` library will not support Python 2 from 2019.
See `this link <http://pymatgen.org/#py3k-only-with-effect-from-2019-1-1>`_ for details.
We are also planing to drop python 3.5 support at feature because we noticed that from version 0.24.0, `pandas <https://pandas.pydata.org/>`_ has no support on python 3.5.

Also, note that XenonPy uses PyTorch_ to accelerate the neural network model training.
If you install XenonPy with PyTorch in windows os, additional tools will be needed.
We highly recommend you to install the `Visual C++ Build Tools <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_ before installing our packages.
Through the installation screen of Visual C++ Build Tools, you need to check the **windows 8.1 / 10 SDK options**.


.. _install_xenonpy:

---------
Using pip
---------

pip is a package management system for installing and updating Python packages,
which comes with any Python installation. On Ubuntu and Fedora Linux,
please use the system package manager to install the ``python3-pip`` package.
We suggest you to `upgrade pip <https://pip.pypa.io/en/stable/installing/>`_ before using it to install other packages.

Before installing XenonPy, You need to install the peer dependencies PyTorch, pymatgen and rdkit_ first.
The easiest way to install all 3 packages is to use conda_.
The following official tutorials will guide you through a successful installation.

PyTorch: https://pytorch.org/get-started/locally/
:raw-html:`<br />`
pymatgen: http://pymatgen.org/index.html#getting-pymatgen
:raw-html:`<br />`
rdkit: https://www.rdkit.org/docs/Install.html

We also provide some preset environments at `conda_env <https://github.com/yoshida-lab/XenonPy/tree/master/conda_env>`_.
If you use linux or mac, you can use these files to build your running environment directly.

1. Chose a environment file and download it. For example the ``xepy36_cuda10.yml``.

.. code-block:: bash

    $ curl -O https://raw.githubusercontent.com/yoshida-lab/XenonPy/master/conda_env/xepy36_cuda10.yml

2. Build the environment from download file. The following commands will build a conda environment named **xepy36_cuda10**.

.. code-block:: bash

    $ conda env create -f xepy36_cuda10.yml

3. Enter the environment **xepy36_cuda10**. Use

.. code-block:: bash

    $ source activate xepy36_cuda10

or

.. code-block:: bash

    $ conda activate xepy36_cuda10

based on the configuration of your conda installation.

When you reached this point, the remaining steps are very simple.
The following command will install XenonPy into your python environment.

.. code-block:: bash

    $ pip install xenonpy

Users can use the following command to install the package at a user-specified directory.

.. code-block:: bash

    $ pip install xenonpy --user

The pre-installed version could be updated to the latest stable release as follow.

.. code-block:: bash

    $ pip install --upgrade xenonpy


------------
Using docker
------------

.. image:: _static/docker.png


**Docker** is a tool designed to easily create, deploy, and run applications across multiple platforms using containers.
Containers allow a developer to pack up an application with all of the parts it needs, such as libraries and other dependencies, into a single package.
We provide the `official docker images`_ via the `Docker hub <https://hub.docker.com>`_.

If you have not installed Docker yet, follow the `official installation tutorial <https://docs.docker.com/install/>`_ to install docker CE on your machine.
Once your docker installation is done, use the following command to boot up a jupyterlab_ with XenonPy available out-of-the-box.

.. code-block:: bash

    $ docker run --rm -it -v $HOME/.xenonpy:/home/user/.xenonpy -v <path/to/your/work_space>:/workspace -p 8888:8888 yoshidalab/xenonpy

Then, open http://localhost:8888 from your favourite browser.

If you have a GPU server/PC running linux and want to bring the GPU acceleration to docker. Just adding ``--runtime=nvidia`` to ``docker run`` command.

.. code-block:: bash

    $ docker run --runtime=nvidia --rm -it -v $HOME/.xenonpy:/home/user/.xenonpy -v <path/to/your/work_space>:/workspace -p 8888:8888 yoshidalab/xenonpy

For more information about **use GPU acceleration in docker**, see `nvidia docker <https://github.com/NVIDIA/nvidia-docker>`_.


permission failed
-----------------

Because docker is a container system running like a virtual machine.
You may face some permission problem when you try to open/save your jupyter files in docker.

The simplest way to resolve these problem is changing the permission of failed files.
You can open a terminal in jupyter notebook and typing:

.. code-block:: bash

    $ sudo chmod 666 permission_failed_file

This will change file permission to ``r+w`` for all users.


------------------------------
Installing in development mode
------------------------------

To use the latest development version distributed at `Github repository`_,
just clone the repository to create a local copy:

.. code-block:: bash

    $ git clone https://github.com/yoshida-lab/XenonPy.git

under the cloned folder, run the following to install XenonPy in development mode:

.. code-block:: bash

    $ cd XenonPy
    $ pip install -e .

To update XenonPy, use ``git fetch && git pull`` 

.. code-block:: bash

    $ git fetch && git pull



----------------------
Troubleshooting/issues
----------------------

Contact us at issues_ and Gitter_ when you have a trouble.

Please provide detailed information (system specification, Python version, and input/output log, and so on).

-----------------------------------------------------------------------------------------------------------

.. _conda: http://conda.pydata.org
.. _official docker images: https://cloud.docker.com/u/yoshidalab/repository/docker/yoshidalab/xenonpy
.. _yoshida-lab channel: https://anaconda.org/yoshida
.. _pip: https://pip.pypa.io
.. _docker image: https://docs.docker.com
.. _Github repository: https://github.com/yoshida-lab/XenonPy
.. _issues: https://github.com/yoshida-lab/XenonPy/issues
.. _Gitter: https://gitter.im/yoshida-lab/XenonPy
.. _PyTorch: http://pytorch.org/
.. _rdkit: https://www.rdkit.org/
.. _jupyterlab: https://jupyterlab.readthedocs.io/en/stable/
