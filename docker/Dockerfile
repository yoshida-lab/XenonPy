FROM nvidia/cuda:10.0-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libxrender1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it
# All users can use /home/user as their home directory
RUN adduser --disabled-password --gecos '' --shell /bin/bash user &&\
    echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user &&\
    mkdir /workspace && chown -R user:user /workspace &&\
    chmod 777 -R /home/user && chmod 777 -R /workspace
USER user
ENV HOME=/home/user
WORKDIR /workspace


# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p ~/miniconda \
    && rm ~/miniconda.sh \
    && /home/user/miniconda/bin/conda update -n base -c defaults conda
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
    && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
    && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# install ppytorch 1.0.0 with CUDA 10.0
# use CUDA 10.0-specific steps
RUN conda install -y -c pytorch \
    cuda100=1.0 \
    magma-cuda100=2.4.0 \
    "pytorch=1.0.0=py3.6_cuda10.0.130_cudnn7.4.1_1" \
    torchvision=0.2.1 &&\
    python -c "import torch; print('pytorch %s' % torch.__version__)"

# install rdkit and pymatgen
RUN conda install -y -c rdkit rdkit=2018.09.1 &&\
    python -c "import rdkit; print('rdkit %s' % rdkit.__version__); from rdkit import Chem"


# install other packages
RUN conda install -y \
    jupyter=1.0.0 \
    scikit-learn=0.20.1 \
    scipy=1.1.0 \
    pandas=0.23.4 \
    requests \
    plotly \
    seaborn \
    pytest=4.0.2 \
    pytest-cov=2.6.0 \
    && conda clean -ya \
    && pip install \
    jupyterlab \
    pybtex \
    openpyxl \
    pymatgen==2018.12.12 \
    matminer &&\
    sudo apt-get purge gcc  -y\
    && sudo apt-get autoremove -y \
    && sudo apt-get clean -y \ 
    && sudo rm -rf /var/lib/apt/lists/* &&\
    python -c "import numpy; print('numpy %s' % numpy.__version__)" &&\
    python -c "import scipy; print('scipy %s' % scipy.__version__)" &&\
    python -c "import pandas; print('pandas %s' % pandas.__version__)" &&\
    python -c "import pymatgen; print('pymatgen %s' % pymatgen.__version__)"


# allow matplotlib working in docker
WORKDIR /etc
RUN echo "backend : Agg" | sudo tee -a matplotlibrc 

WORKDIR /opt/xenonpy
COPY . .
RUN sudo chown -R user:user /opt && find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf && \
    pip install --user -v .  && pytest tests -v

EXPOSE 8888

WORKDIR /workspace
CMD [ "jupyter" , "lab", "--ip=0.0.0.0", "--no-browser", "--port=8888", "--allow-root"]