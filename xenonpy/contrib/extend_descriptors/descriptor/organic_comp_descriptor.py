#  Copyright (c) 2019. stewu5. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.e

from collections import Counter

import pandas as pd
from rdkit import Chem
from xenonpy.descriptor import Compositions
from xenonpy.descriptor.base import BaseFeaturizer


class OrganicCompDescriptor(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, featurizers='all', on_errors='raise', return_type='any'):
        """
        A featurizer for extracting XenonPy compositional descriptors from SMILES or MOL
        """
            
        # fix n_jobs to be 0 to skip automatic wrapper in XenonPy BaseFeaturizer class
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)
        self._cal = Compositions(n_jobs=n_jobs, featurizers=featurizers, on_errors=on_errors)

    def featurize(self, x):
        # check if type(x) = list
        if isinstance(x, pd.Series):
            x = x.tolist()
        if not isinstance(x, list):
            x = [x]
        # check input format, assume SMILES if not RDKit-MOL
        if not isinstance(x[0], Chem.rdchem.Mol):
            x_mol = []
            for z in x:
                x_mol.append(Chem.MolFromSmiles(z))
                if x_mol[-1] is None:
                    raise ValueError('can not convert Mol from SMILES %s' % z)
        else:
            x_mol = x
        
        # convert to counting dictionary
        mol = [Chem.AddHs(z) for z in x_mol]
        d_list = [dict(Counter([atom.GetSymbol() for atom in z.GetAtoms()])) for z in mol]

        self.output = self._cal.transform(d_list)
        
        return self.output
    
    @property
    def feature_labels(self):
        return self.output.columns
