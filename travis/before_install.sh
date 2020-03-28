#!/usr/bin/env bash

# This script is meant to be called by the "before_install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

export MPLBACKEN='Agg'

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Download miniconda for osx
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh

else
    # Download miniconda for Linux
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
fi

# Install miniconda
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"

# conda info
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

# Create conda env from environment files
case "${TRAVIS_OS_NAME}" in
    osx)
        conda create -n tests python=${PYENV}
        conda env update -n tests -f travis/osx_env.yml
        source activate tests
        ;;
    linux)
        conda create -n tests python=${PYENV}
        conda env update -n tests -f travis/linux_win_env.yml
        source activate tests
        ;;
esac

conda info --envs
pip install codecov
conda list

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import torch; print('pytorch %s' % torch.__version__)"
python -c "import pymatgen; print('pymatgen %s' % pymatgen.__version__)"
python -c "import rdkit; print('rdkit %s' % rdkit.__version__)"
python -c "from rdkit import Chem; print(Chem)"

