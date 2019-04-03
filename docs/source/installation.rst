.. role:: raw-html(raw)
    :format: html

============
Installation
============

XenonPy can be installed using pip_ in 3.6 and 3.7 on Mac, Linux and Windows.
Alternatively, we recommend using the `Docker Image`_ if you have no installation preference.
We have no plan to support Python 2.x. One of the main reasons is that the ``pymatgen`` library will not support Python 2 from 2019.
See `this link <http://pymatgen.org/#py3k-only-with-effect-from-2019-1-1>`_ for details.



.. _install_xenonpy:

-------------------
Using conda and pip
-------------------

pip_ is a package management system for installing and updating Python packages, which comes with any Python distribution.
Conda_ is an open source package management system and environment management system that runs on Windows, macOS and Linux.
Conda easily creates, saves, loads and switches between environments on your local computer.

XenonPy has 3 peer dependencies, which are PyTorch, pymatgen and rdkit_. Before you install XenonPy, you have to install them.
The easiest way to install all 3 packages is to use Conda_. The following official tutorials will guide you through a successful installation.

PyTorch: https://pytorch.org/get-started/locally/
:raw-html:`<br />`
pymatgen: http://pymatgen.org/index.html#getting-pymatgen
:raw-html:`<br />`
rdkit: https://www.rdkit.org/docs/Install.html

For convenience, several environments preset are available at `conda_env <https://github.com/yoshida-lab/XenonPy/tree/master/conda_env>`_.
You can use these files to create the runnable environment on locally.

.. code-block:: bash

    $ conda env create -f <path_to_file>

For Unix-like (linux, mac, FreeBSD, etc.) system, the above command will be enough.
For windows, additional tools are needed. We highly recommend you to install the `Visual C++ Build Tools <http://landinghub.visualstudio.com/visual-cpp-build-tools>`_ before creating your environment.
Also, confirm that you have checked the **windows 8.1 / 10 SDK options** on when installing the build tools.

The following example shows how to create an environment step-by-step.

1. **Chose an environment and download the corresponding configuration file**.

 First, choose a configuration preset from `here <https://github.com/yoshida-lab/XenonPy/tree/master/conda_env>`_, then use it to create the runtime environment locally.
 For example, you want to run XenonPy in python 3.6 with cuda10 support. The ``xepy36_cuda10.yml`` will be satisfied.
 In Unix-like system with `curl <https://curl.haxx.se/>`_ has been installed, the following command will download the configuration file and save it locally.

 .. code-block:: bash

  $ curl -O https://raw.githubusercontent.com/yoshida-lab/XenonPy/master/conda_env/xepy36_cuda10.yml

2. **Create the environment from file**.

 Run the following commands to build the conda environment based on the configuration file.

 .. code-block:: bash

    $ conda env create -f xepy36_cuda10.yml


3. **Enter the environment**.

 The name of environment is the same as the configuration file. For our example is  **xepy36_cuda10**.

 .. code-block:: bash

    $ source activate xepy36_cuda10

    # or

    $ activate xepy36_cuda10

    # or

    $ conda activate xepy36_cuda10

 Which command should be used is based on your system and your conda configuration.

4. **Install XenonPy**

 When you reached this point, the remaining steps are very simple.
 Using ``pip install xenonpy`` to install XenonPy into the environment.

 .. code-block:: bash

    $ pip install xenonpy

 Also, you can give ``--user`` option to ``pip install`` to install a user-specified directory.

 .. code-block:: bash

    $ pip install xenonpy --user

 Last, old version could be updated as follow.

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

.. _Conda: https://conda.io/en/latest/
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
