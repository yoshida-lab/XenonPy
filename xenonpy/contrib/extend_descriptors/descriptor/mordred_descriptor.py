#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from mordred import Calculator, descriptors
from rdkit import Chem

from xenonpy.descriptor.base import BaseFeaturizer


class Mordred2DDescriptor(BaseFeaturizer):

    def __init__(self, *, on_errors='raise', return_type='any'):
        # fix n_jobs to be 0 to skip automatic wrapper in XenonPy BaseFeaturizer class
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)
        self.output = None
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        # check if type(x) = list
        if not isinstance(x, (list,)):
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

        calc = Calculator(descriptors, ignore_3D=True)
        self.output = calc.pandas(x_mol)
        return self.output

    @property
    def feature_labels(self):
        return self.output.columns
