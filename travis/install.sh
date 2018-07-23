#!/usr/bin/env bash

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
export MPLBACKEN='Agg'

hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
# Useful for debugging any issues with conda
conda info -a

if [[ "$PYTHON_VERSION" == "3.5" ]]; then
    conda env create -f travis/environment_py35.yml
    source activate xepy35
elif [[ "$PYTHON_VERSION" == "3.6" ]]; then
    conda env create -f travis/environment_py36.yml
    source activate xepy36
else
    conda env create -f travis/environment_py37.yml
    source activate xepy36
fi

conda install pytest pytest-cov pylint

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"


#python setup.py install