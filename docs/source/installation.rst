.. role:: raw-html(raw)
    :format: html

============
Installation
============

XenonPy can be installed using pip_ in Python 3.5, 3.6 and 3.7 on Mac, Linux and Windows.
Alternately, we recommend you use the `Docker Image`_ if you have no installation preference.

We have no plane to support Python 2.x. One of the main reasons is that the ``pymatgen`` library will discontinuing Python 2 support from 2019.
See this `this link <http://pymatgen.org/#py3k-only-with-effect-from-2019-1-1>`_ for details.

Also note that XenonPy use PyTorch_ to accelerate the neural network model training.
If you instal XenonPy with PyTorch in windows os, some additional tools will be needed.
We are highly recommend that you install the `Visual C++ Build Tools <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_ before the package installing.
Also, through the installation screen of Visual C++ Build Tools, you need to check-on the **windows 8.1 / 10 SDK options**.


.. _install_xenonpy:

Use pip
=======

pip is a package management system for installing and updating Python packages,
and comes with any Python installation. On Ubuntu and Fedora Linux,
use the system package manager to install the ``python3-pip`` package.
We suggest you `upgrade pip <https://pip.pypa.io/en/stable/installing/>`_ before using it to install other programs.

Before install XenonPy, You need to install the peerdependencies PyTorch, pymatgen and rdkit_ first.
The easiest way to install the 3 packages together is to use conda_.
The following official tutorials will leading you to a successful installation.

PyTorch: https://pytorch.org/get-started/locally/
:raw-html:`<br />`
pymatgen: http://pymatgen.org/index.html#getting-pymatgen
:raw-html:`<br />`
rdkit: https://www.rdkit.org/docs/Install.html

When you done, the remaining is very simple.
The following command install XenonPy in your default python environment.

.. code-block:: bash

    $ pip install xenonpy

Users can use this command to install at a user-specified directory:

.. code-block:: bash

    $ pip install xenonpy --user

The pre-installed version could be renewed to the latest stable release as

.. code-block:: bash

    $ pip install --upgrade xenonpy


Use docker
==========

.. image:: _static/docker.png


Docker is a tool designed to make it easier to create, deploy, and run applications by using containers.
Containers allow a developer to package up an application with all of the parts it needs, such as libraries and other dependencies, and ship it all out as one package.
We provide the `official docker images`_ via the `Docker hub <https://hub.docker.com>`_.

If you have not installed Docker yet, Follow the `official install tutorial <https://docs.docker.com/install/>`_ to install docker CE on your machine.
Once your docker installation is done, use the following command to boot up XenonPy with jupyterlab_.

.. code-block:: bash

    $ docker run --rm -it -v $HOME/.xenonpy:/root/.xenonpy -v <path/to/your/work_space>:/root -p 8888:8888 yoshidalab/xenonpy

Then open http://localhost:8888 from your favourite browser.

Install in development mode
===========================

To use the latest development version distributed at `Github repository`_,
just clone the repository to create a local copy:

.. code-block:: bash

    $ git clone https://github.com/yoshida-lab/XenonPy.git

under the cloned folder, run the following to install XenonPy in development mode:

.. code-block:: bash

    $ cd XenonPy
    $ pip install -e .

To update XenonPy, use ``git fetch $$ git pull`` 

.. code-block:: bash

    $ git fetch $$ git pull



Troubleshooting/Issues
======================

Contact us at issues_ and Gitter_ when you have a trouble.

Please provide fully detailed information (system specification, Python version, and input/output log, and so on).

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