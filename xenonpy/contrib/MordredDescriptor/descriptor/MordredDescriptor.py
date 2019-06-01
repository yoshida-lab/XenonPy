# XenonPy BaseFeaturizer for calculating 2D Mordred descriptors
# n_jobs is set to 0 to use the default batch calculator in the Mordred package

from xenonpy.descriptor.base import BaseFeaturizer
from mordred import Calculator, descriptors

class Mordred2DDescriptor(BaseFeaturizer):

    def __init__(self, *, on_errors='raise', return_type='any'):
        # fix n_jobs to be 0 to skip automatic wrapper in XenonPy BaseFeaturizer class
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)

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
