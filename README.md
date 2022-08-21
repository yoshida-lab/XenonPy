<p align="center">
  <img height="200" src="https://github.com/yoshida-lab/XenonPy/blob/master/logo.png" alt="xenonpy">
</p>

# NOTES

> **To all those who have purchased the book [マテリアルズインフォマティクス](https://www.kyoritsu-pub.co.jp/book/b10013510.html) published by [KYORITSU SHUPPAN](https://www.kyoritsu-pub.co.jp/): The link to the exercises has changed to https://github.com/yoshida-lab/XenonPy/tree/master/mi_book. Please follow the new link to access all these exercises.**

> **Our XenonPy.MDL is under technical maintenance due to some security issues found during a server upgrade. Our current plan is to recover the access at the release of v0.7 (a conservative estimate of release date would be Sep. 2022). In the mean time, if you would like to get access to the pretrained models, please contact us directly with your purpose of using the models and your affiliation. We will try to provide necessary aid to access part of the model library based on specific needs. Sorry for all the inconvenience. We will make further announcement here when a more concrete recovery schedule is available.**


**We apologize for the inconvenience** 🥺🙏🙇

# XenonPy project

[![MacOS](https://github.com/yoshida-lab/XenonPy/workflows/MacOS/badge.svg)](https://github.com/yoshida-lab/XenonPy/actions?query=workflow%3AMacOS)
[![Windows](https://github.com/yoshida-lab/XenonPy/workflows/Windows/badge.svg)](https://github.com/yoshida-lab/XenonPy/actions?query=workflow%3AWindows)
[![Ubuntu](https://github.com/yoshida-lab/XenonPy/workflows/Ubuntu/badge.svg)](https://github.com/yoshida-lab/XenonPy/actions?query=workflow%3AUbuntu)
[![Documentation Status](https://readthedocs.org/projects/xenonpy/badge/?version=latest)](https://xenonpy.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/yoshida-lab/XenonPy/branch/master/graph/badge.svg)](https://codecov.io/gh/yoshida-lab/XenonPy)
[![Version](https://img.shields.io/github/tag/yoshida-lab/XenonPy.svg?maxAge=360)](https://github.com/yoshida-lab/XenonPy/releases/latest)
[![Python Versions](https://img.shields.io/pypi/pyversions/xenonpy.svg)](https://pypi.org/project/xenonpy/)
[![Downloads](https://pepy.tech/badge/xenonpy)](https://pepy.tech/project/xenonpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/xenonpy.svg?label=PiPy%20downloads)

**XenonPy** is a Python library that implements a comprehensive set of machine learning tools
for materials informatics. Its functionalities partially depend on PyTorch and R.
The current release provides some limited modules:

-   Interface to public materials database
-   Library of materials descriptors (compositional/structural descriptors)
-   Pre-trained model library **XenonPy.MDL** (v0.1.0.beta, 2019/8/7: more than 140,000 models (include private models) in 35 properties of small molecules, polymers, and inorganic compounds)
-   Machine learning tools.
-   Transfer learning using the pre-trained models in XenonPy.MDL

XenonPy inspired by matminer: https://hackingmaterials.github.io/matminer/.

XenonPy is a open source project https://github.com/yoshida-lab/XenonPy.

See our documents for details: http://xenonpy.readthedocs.io

## Publications

1. H. Ikebata, K. Hongo, T. Isomura, R. Maezono, and R. Yoshida, “Bayesian molecular design with a chemical language model,” J Comput Aided Mol Des, vol. 31, no. 4, pp. 379–391, Apr. 2017, doi: 10/ggpx8b.
2. S. Wu et al., “Machine-learning-assisted discovery of polymers with high thermal conductivity using a molecular design algorithm,” npj Computational Materials, vol. 5, no. 1, pp. 66–66, Dec. 2019, doi: 10.1038/s41524-019-0203-2.
3. S. Wu, G. Lambard, C. Liu, H. Yamada, and R. Yoshida, “iQSPR in XenonPy: A Bayesian Molecular Design Algorithm,” Mol. Inform., vol. 39, no. 1–2, p. 1900107, Jan. 2020, doi: 10.1002/minf.201900107.
4. H. Yamada et al., “Predicting Materials Properties with Little Data Using Shotgun Transfer Learning,” ACS Cent. Sci., vol. 5, no. 10, pp. 1717–1730, Oct. 2019, doi: 10.1021/acscentsci.9b00804.
5. S. Ju et al., “Exploring diamondlike lattice thermal conductivity crystals via feature-based transfer learning,” Phys. Rev. Mater., vol. 5, no. 5, p. 053801, May 2021, doi: 10.1103/physrevmaterials.5.053801.
6. C. Liu et al., “Machine Learning to Predict Quasicrystals from Chemical Compositions,” Adv. Mater., vol. 33, no. 36, p. 2102507, Sep. 2021, doi: 10.1002/adma.202102507.

## XenonPy images (deprecated)

> Docker has introduced a new [Subscription Service Agreement](https://www.docker.com/legal/docker-subscription-service-agreement) which requires organizations with more than 250 employees or more than $10 million in revenue to buy a paid subscription.
> Since the fact that Docker company has been changed their policy to business first mode, we decided to drop the prebuilt Docker images service.

[XenonPy base images](https://hub.docker.com/repository/docker/yoshidalab/base) packed a lot of useful packages for materials informatics using.
The following table list some core packages in XenonPy images.

| Package        | Version   |
| -------------- | --------- |
| `PyTorch`      | 1.7.1     |
| `tensorly`     | 0.5.0     |
| `pymatgen`     | 2021.2.16 |
| `matminer`     | 0.6.2     |
| `mordred`      | 1.2.0     |
| `scipy`        | 1.6.0     |
| `scikit-learn` | 0.24.1    |
| `xgboost`      | 1.3.0     |
| `ngboost`      | 0.3.7     |
| `fastcluster`  | 1.1.26    |
| `pandas`       | 1.2.2     |
| `rdkit`        | 2020.09.4 |
| `jupyter`      | 1.0.0     |
| `seaborn`      | 0.11.1    |
| `matplotlib`   | 3.3.4     |
| `OpenNMT-py`   | 1.2.0     |
| `Optuna`       | 2.3.0     |
| `plotly`       | 4.11.0    |
| `ipympl`       | 0.5.8     |

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
you can pull the CUDA 10.1 version with:

```bash
docker pull yoshidalab/xenonpy:cuda10
```

The table below lists software versions for each of the currently supported
Docker image tags .

| Image tag | CUDA | PyTorch |
| --------- | ---- | ------- |
| `latest`  | 11.0 | 1.7.1   |
| `cpu`     | None | 1.7.1   |
| `cuda11`  | 11.0 | 1.7.1   |
| `cuda10`  | 10.2 | 1.7.1   |
| `cuda9`   | 9.2  | 1.7.1   |

### Running XenonPy

It is possible to run XenonPy inside a container.
Using xenonpy with jupyter is very easy, you could run it with
the following command:

```sh
docker run --rm -it \
  --runtime=nvidia \
  --ipc=host \
  --publish="8888:8888" \
  --volume=$HOME/.xenonpy:/home/user/.xenonpy \
  --volume=<path/to/your/workspace>:/workspace \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  yoshidalab/xenonpy
```

Here's a description of the Docker command-line options shown above:

-   `--runtime=nvidia`: Required if using CUDA, optional otherwise. Passes the
    graphics card from the host to the container. **Optional, based on your usage**.
-   `--ipc=host`: Required if using multiprocessing, as explained at
    <https://github.com/pytorch/pytorch#docker-image.> **Optional**
-   `--publish="8888:8888"`: Publish container's port 8888 to the host. **Needed**
-   `--volume=$Home/.xenonpy:/home/user/.xenonpy`: Mounts
    the XenonPy root directory into the container. **Optional, but highly recommended**.
-   `--volume=<path/to/your/workspace>:/workspace`: Mounts
    the your working directory into the container. **Optional, but highly recommended**.
-   `-e NVIDIA_VISIBLE_DEVICES=0`: Sets an environment variable to restrict which
    graphics cards are seen by programs running inside the container. Set to `all`
    to enable all cards. Optional, defaults to all.

You may wish to consider using [Docker Compose](https://docs.docker.com/compose/)
to make running containers with many options easier. At the time of writing,
only version 2.3 of Docker Compose configuration files supports the `runtime`
option.

## Copyright and license

©Copyright 2021 The XenonPy project, all rights reserved.
Released under the `BSD-3 license`.
