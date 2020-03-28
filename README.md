<p align="center">
  <img height="200" src="https://github.com/yoshida-lab/XenonPy/blob/master/logo.png" alt="xenonpy">
</p>

# XenonPy project

[![Build Status](https://travis-ci.org/yoshida-lab/XenonPy.svg?branch=master)](https://travis-ci.org/yoshida-lab/XenonPy)
[![Build Status](https://api.cirrus-ci.com/github/yoshida-lab/XenonPy.svg?branch=master)](https://cirrus-ci.com/github/yoshida-lab/XenonPy)
[![Build status](https://ci.appveyor.com/api/projects/status/vnh350xqffp6t9nk/branch/master?svg=true)](https://ci.appveyor.com/project/TsumiNa/xenonpy/branch/master)
[![codecov](https://codecov.io/gh/yoshida-lab/XenonPy/branch/master/graph/badge.svg)](https://codecov.io/gh/yoshida-lab/XenonPy)
[![Version](https://img.shields.io/github/tag/yoshida-lab/XenonPy.svg?maxAge=360)](https://github.com/yoshida-lab/XenonPy/releases/latest)
[![Python Versions](https://img.shields.io/pypi/pyversions/xenonpy.svg)](https://pypi.org/project/xenonpy/)
[![Downloads](https://pepy.tech/badge/xenonpy)](https://pepy.tech/project/xenonpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/xenonpy.svg?label=PiPy%20downloads)

**XenonPy** is a Python library that implements a comprehensive set of machine learning tools
for materials informatics. Its functionalities partially depend on PyTorch and R.
The current release provides some limited modules:

- Interface to public materials database
- Library of materials descriptors (compositional/structural descriptors)
- Pre-trained model library **XenonPy.MDL** (v0.1.0.beta, 2019/8/7: more than 140,000 models (include private models) in 35 properties of small molecules, polymers, and inorganic compounds)
- Machine learning tools.
- Transfer learning using the pre-trained models in XenonPy.MDL

XenonPy inspired by matminer: https://hackingmaterials.github.io/matminer/.

XenonPy is a open source project https://github.com/yoshida-lab/XenonPy.

See our documents for details: http://xenonpy.readthedocs.io


## Publications
1. Yamada, H. et al. Predicting Materials Properties with Little Data Using Shotgun Transfer Learning. ACS Cent. Sci. acscentsci.9b00804 (2019). doi:10.1021/acscentsci.9b00804

## XenonPy images

XenonPy images packed a lot of useful packages for materials informatics using.
The following table list some core packages in XenonPy images.

| Package        | Version    |
| -------------- | ---------- |
| `PyTorch`      | 1.4.0      |
| `tensorly`     | 0.4.4      |
| `pymatgen`     | 2020.1.28  |
| `matminer`     | 0.6.2      |
| `mordred`      | 1.2.0      |
| `scipy`        | 1.3.1      |
| `scikit-learn` | 0.22.1     |
| `xgboost`      | 1.0.0      |
| `ngboost`      | master     |
| `pandas`       | 1.0.0      |
| `rdkit`        | 2019.03.3  |
| `jupyter`      | 1.0.0      |
| `seaborn`      | 0.9.0      |
| `matplotlib`   | 3.1.2      |
| `plotly`       | 4.5.0      |

## Requirements

In order to use this image you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

### CUDA requirements

If you have a CUDA-compatible NVIDIA graphics card, you can use a CUDA-enabled
version of the PyTorch image to enable hardware acceleration. This **only** can be
used in Ubuntu Linux.

Firstly, ensure that you install the appropriate NVIDIA drivers and libraries.
If you are running Ubuntu, you can install proprietary NVIDIA drivers
[from the PPA](https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa)
and CUDA [from the NVIDIA website](https://developer.nvidia.com/cuda-downloads).

You will also need to install `nvidia-docker2` to enable GPU device access
within Docker containers. This can be found at
[NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

## Usage

Pre-built xenonpy images are available on Docker Hub under the name
[yoshidalab/xenonpy](https://hub.docker.com/r/yoshidalab/xenonpy/). For example,
you can pull the CUDA 10.0 version with:

```bash
docker pull yoshidalab/xenonpy:cuda10
```

The table below lists software versions for each of the currently supported
Docker image tags .

| Image tag | CUDA | PyTorch |
| --------- | ---- | ------- |
| `latest`  | 10.1 | 1.4.0   |
| `cpu`     | None | 1.4.0   |
| `cuda10`  | 10.1 | 1.4.0   |
| `cuda9`   | 9.2  | 1.4.0   |

### Running XenonPy

It is possible to run XenonPy inside a container.
Using xenonpy with jupyter is very easy, you could run it with
the following command:

```sh
docker run --rm -it \
  --runtime=nvidia \
  --ipc=host \
  --publish="8888:8888"
  --volume=$Home/.xenonpy:/home/user/.xenonpy \
  --volume=<path/to/your/workspace>:/workspace \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  yoshidalab/xenonpy
```

Here's a description of the Docker command-line options shown above:

- `--runtime=nvidia`: Required if using CUDA, optional otherwise. Passes the
  graphics card from the host to the container. **Optional, based on your usage**.
- `--ipc=host`: Required if using multiprocessing, as explained at
  <https://github.com/pytorch/pytorch#docker-image.> **Optional**
- `--publish="8888:8888"`: Publish container's port 8888 to the host. **Needed**
- `--volume=$Home/.xenonpy:/home/user/.xenonpy`: Mounts
  the XenonPy root directory into the container. **Optional, but highly recommended**.
- `--volume=<path/to/your/workspace>:/workspace`: Mounts
  the your working directory into the container. **Optional, but highly recommended**.
- `-e NVIDIA_VISIBLE_DEVICES=0`: Sets an environment variable to restrict which
  graphics cards are seen by programs running inside the container. Set to `all`
  to enable all cards. Optional, defaults to all.

You may wish to consider using [Docker Compose](https://docs.docker.com/compose/)
to make running containers with many options easier. At the time of writing,
only version 2.3 of Docker Compose configuration files supports the `runtime`
option.

## Copyright and license

©Copyright 2020 The XenonPy project, all rights reserved.
Released under the `BSD-3 license`.
