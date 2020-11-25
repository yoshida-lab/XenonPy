#!/usr/bin/env bash

cwd
ll -al

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import torch; print('pytorch %s' % torch.__version__)"
python -c "import pymatgen; print('pymatgen %s' % pymatgen.__version__)"
python -c "import rdkit; print('rdkit %s' % rdkit.__version__)"
python -c "from rdkit import Chem; print(Chem)"
