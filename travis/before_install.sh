#!/usr/bin/env bash

# This script is meant to be called by the "before_install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.

export MPLBACKEN='Agg'

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then

    # Install some custom requirements on osx
    brew update
    brew upgrade
    brew cask install miniconda

else
    # Install some custom requirements on Linux
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# conda test info
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a


case "${PYENV}" in
    py35)
        conda env create -f travis/environment_py35.yml
        source activate xepy35
        ;;
    py36)
        conda env create -f travis/environment_py36.yml
        source activate xepy36
        ;;
    py37)
        conda env create -f travis/environment_py36.yml
        source activate xepy36
        ;;
esac

conda install pytest pytest-cov pylint

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import torch; print('pandas %s' % torch.__version__)"

#python setup.py install