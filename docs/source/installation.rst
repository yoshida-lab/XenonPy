.. role:: raw-html(raw)
    :format: html

============
Installation
============

XenonPy can be installed using pip_ in 3.6 and 3.7 on Mac, Linux, and Windows.
Alternatively, we recommend using the `Docker Image`_ if you have no installation preference.
We have no plan to support Python 2.x. One of the main reasons is that the ``pymatgen`` library will not support Python 2 from 2019.
See `this link <http://pymatgen.org/#py3k-only-with-effect-from-2019-1-1>`_ for details.



.. _install_xenonpy:

-------------------
Using conda and pip
-------------------

pip_ is a package management system for installing and updating Python packages, which comes with any Python distribution.
Conda_ is an open source package management system and environment management system that runs on Windows, macOS, and Linux.
Conda easily creates, saves, loads and switches between environments on your local computer.

XenonPy has 3 peer dependencies, which are PyTorch, pymatgen and rdkit_. Before you install XenonPy, you have to install them.
The easiest way to install all 3 packages is to use Conda_. The following official tutorials will guide you through a successful installation.

PyTorch: https://pytorch.org/get-started/locally/
:raw-html:`<br />`
pymatgen: http://pymatgen.org/index.html#getting-pymatgen
:raw-html:`<br />`
rdkit: https://www.rdkit.org/docs/Install.html

For convenience, several environments preset are available at `conda_env`_.
You can use these files to create a runnable environment on your local machine.

.. code-block:: bash

    $ conda create -n <env_name_you_liked> python={3.6 or 3.7}  # use `conda create` command to create a fresh environment with specific name and python version
    $ conda env update -n <env_name_you_created> -f <path_to_env_file>  # use `conda env update` command to sync packages with the preset environment

.. note::

    For Unix-like (Linux, Mac, FreeBSD, etc.) system, the above command will be enough. For windows, additional tools are needed.
    We highly recommend you to install the `Visual C++ Build Tools <https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16>`_ before creating your environment.
    Also, confirm that you have checked the **Windows 10 SDK** (assuming the computer is Windows 10) on when installing the build tools.

The following example shows how to create an environment named ``xenonpy`` in ``python3.7`` and ``cuda10`` supports step-by-step.

1. **Chose an environment preset and download it**.

 Because we want to have cuda10 supports, The **cuda10.yml** preset will surely satisfy this work.
 If `curl <https://curl.haxx.se/>`_ has been installed, the following command will download the environment file to your local machine.

 .. code-block:: bash

  $ curl -O https://raw.githubusercontent.com/yoshida-lab/XenonPy/master/conda_env/cuda10.yml

2. **Create the environment and install packages using environment file**.

 The following commands will rebuild a python3.7 environment, and install the packages which are listed in **cuda10.yml**.

 .. code-block:: bash

    $ conda create -n xenonpy python=3.7
    $ conda env update -n xenonpy -f cuda10.yml


3. **Enter the environment by name**.

 In this example, it's ``xenonpy``.

 .. code-block:: bash

    $ source activate xenonpy

    # or

    $ activate xenonpy

    # or

    $ conda activate xenonpy

 .. note::
     Which command should be used is based on your system and your conda configuration.

4. **Update XenonPy**

 When you reached here, XenonPy has been installed successfully.
 If you want to update your old installation of XenonPy, ssing ``pip install -U xenonpy``.

 .. code-block:: bash

    $ pip install -U xenonpy


------------
Using docker
------------

.. image:: _static/docker.png


**Docker** is a tool designed to easily create, deploy, and run applications across multiple platforms using containers.
Containers allow a developer to pack up an application with all of the parts it needs, such as libraries and other dependencies, into a single package.
We provide the `official docker images`_ via the `Docker hub <https://hub.docker.com>`_.

Using docker needs you to have a docker installation on your local machine. If you have not installed it yet, follow the `official installation tutorial <https://docs.docker.com/install/>`_ to install docker CE on your machine.
Once you have done this, the following command will boot up a jupyterlab_ for you with XenonPy inside. See `here <https://github.com/yoshida-lab/XenonPy#xenonpy-images>`_ to know what other packages are available.

.. code-block:: bash

    $ docker run --rm -it -v $HOME/.xenonpy:/home/user/.xenonpy -v <path/to/your/work_space>:/workspace -p 8888:8888 yoshidalab/xenonpy

If you have a GPU server/PC running Linux and want to bring the GPU acceleration to docker. Just adding ``--runtime=nvidia`` to ``docker run`` command.

.. code-block:: bash

    $ docker run --runtime=nvidia --rm -it -v $HOME/.xenonpy:/home/user/.xenonpy -v <path/to/your/work_space>:/workspace -p 8888:8888 yoshidalab/xenonpy

For more information about **using GPU acceleration in docker**, see `nvidia docker <https://github.com/NVIDIA/nvidia-docker>`_.


Permission failed
-----------------

You may have a permission problem when you try to open/save jupyter files. This is because docker is a container system running like a virtual machine.
Files will have different permission when be mounted onto a docker container.
The simplest way to resolve this problem is changing the permission of failed files.
You can open a terminal in jupyter notebook and type:

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

Contact us at issues_ and Gitter_ when you have trouble.

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
.. _conda_env: https://github.com/yoshida-lab/XenonPy/tree/master/conda_env
