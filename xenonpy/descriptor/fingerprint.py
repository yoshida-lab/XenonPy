#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys as MAC
from rdkit.Chem import rdMolDescriptors as rdMol
from rdkit.ML.Descriptors import MoleculeDescriptors

from .base import BaseDescriptor, BaseFeaturizer


class RDKitFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, fp_size=2048, input_type='mol', on_errors='raise', return_type='any'):
        """
        RDKit fingerprint.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
        fp_size: int
            Fingerprint size.
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.input_type = input_type
        self.fp_size = fp_size

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('can not convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('can not convert Mol from SMILES %s' % x_)

        return list(Chem.RDKFingerprint(x, fpSize=self.fp_size))

    @property
    def feature_labels(self):
        return ["rdkit:" + str(i) for i in range(self.fp_size)]


class AtomPairFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048, input_type='mol', on_errors='raise', return_type='any'):
        """
        Atom Pair fingerprints.
        Returns the atom-pair fingerprint for a molecule.The algorithm used is described here:
        R.E. Carhart, D.H. Smith, R. Venkataraghavan;
        "Atom Pairs as Molecular Features in Structure-Activity Studies: Definition and Applications"
        JCICS 25, 64-73 (1985).
        This is currently just in binary bits with fixed length after folding.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
        n_bits: int
           Fixed bit length based on folding.
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.input_type = input_type
        self.n_bits = n_bits

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('can not convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('can not convert Mol from SMILES %s' % x_)
        return list(rdMol.GetHashedAtomPairFingerprintAsBitVect(x, nBits=self.n_bits))

    @property
    def feature_labels(self):
        return ['apfp:' + str(i) for i in range(self.n_bits)]


class TopologicalTorsionFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048, input_type='mol', on_errors='raise', return_type='any'):
        """
        Topological Torsion fingerprints.
        Returns the topological-torsion fingerprint for a molecule.
        This is currently just in binary bits with fixed length after folding.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
        n_bits: int
           Fixed bit length based on folding.
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.input_type = input_type
        self.n_bits = n_bits

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('can not convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('can not convert Mol from SMILES %s' % x_)
        return list(rdMol.GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits=self.n_bits))

    @property
    def feature_labels(self):
        return ['ttfp:' + str(i) for i in range(self.n_bits)]


class MACCS(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, input_type='mol', on_errors='raise', return_type='any'):
        """
        The MACCS keys for a molecule. The result is a 167-bit vector. There are 166 public keys,
        but to maintain consistency with other software packages they are numbered from 1.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.input_type = input_type

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('can not convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('can not convert Mol from SMILES %s' % x_)
        return list(MAC.GenMACCSKeys(x))

    @property
    def feature_labels(self):
        return ['maccs:' + str(i) for i in range(167)]


class FCFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048, input_type='mol', on_errors='raise', return_type='any'):
        """
        Morgan (Circular) fingerprints + feature-based (FCFP)
        The algorithm used is described in the paper Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints.
        JCIM 50:742-54 (2010)

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
        radius: int
            The radius parameter in the Morgan fingerprints, which is roughly half of the diameter parameter in FCFP,
            i.e., radius=2 is roughly equivalent to FCFP4.
        n_bits: int
            Fixed bit length based on folding.
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.input_type = input_type
        self.radius = radius
        self.n_bits = n_bits
        # self.arg = arg # arg[0] = radius, arg[1] = bit length

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('can not convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('can not convert Mol from SMILES %s' % x_)
        return list(
            rdMol.GetMorganFingerprintAsBitVect(
                x, self.radius, nBits=self.n_bits, useFeatures=True))

    @property
    def feature_labels(self):
        return ['fcfp3:' + str(i) for i in range(self.n_bits)]


class ECFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048, input_type='mol', on_errors='raise', return_type='any'):
        """
        Morgan (Circular) fingerprints (ECFP)
        The algorithm used is described in the paper Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints.
        JCIM 50:742-54 (2010)

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
        radius: int
            The radius parameter in the Morgan fingerprints, which is roughly half of the diameter parameter in ECFP,
            i.e., radius=2 is roughly equivalent to ECFP4.
        n_bits: int
            Fixed bit length based on folding.
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.input_type = input_type
        self.radius = radius
        self.n_bits = n_bits
        # self.arg = arg # arg[0] = radius, arg[1] = bit length

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('can not convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('can not convert Mol from SMILES %s' % x_)
        return list(rdMol.GetMorganFingerprintAsBitVect(x, self.radius, nBits=self.n_bits))

    @property
    def feature_labels(self):
        return ['ecfp3:' + str(i) for i in range(self.n_bits)]


class DescriptorFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, input_type='mol', on_errors='raise', return_type='any'):
        """
        All descriptors in RDKit (length = 200) [may include NaN]
            see https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors for the full list

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """
        # self.arg = arg # arg[0] = radius, arg[1] = bit length
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)
        self.input_type = input_type
        nms = [x[0] for x in Descriptors._descList]
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('can not convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('can not convert Mol from SMILES %s' % x_)
        return self.calc.CalcDescriptors(x)

    @property
    def feature_labels(self):
        return ['desc200:' + str(i) for i in range(200)]


class Fingerprints(BaseDescriptor):
    """
    Calculate fingerprints or descriptors of organic molecules.
    """

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048, fp_size=2048, input_type='mol', featurizers='all',
                 on_errors='raise'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
        radius: int
            The radius parameter in the Morgan fingerprints,
            which is roughly half of the diameter parameter in ECFP/FCFP,
            i.e., radius=2 is roughly equivalent to ECFP4/FCFP4.
        n_bits: int
            Fixed bit length based on folding.
        featurizers: list[str] or 'all'
            Featurizers that will be used.
            Default is 'all'.
        input_type: string
            Set the specific type of transform input.
            Set to ``mol`` (default) to ``rdkit.Chem.rdchem.Mol`` objects as input.
            When set to ``smlies``, ``transform`` method can use a SMILES list as input.
            Set to ``any`` to use both.
            If input is SMILES, ``Chem.MolFromSmiles`` function will be used inside.
            for ``None`` returns, a ``ValueError`` exception will be raised.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """

        super().__init__(featurizers=featurizers)
        self.n_jobs = n_jobs

        self.mol = RDKitFP(n_jobs, fp_size=fp_size, input_type=input_type, on_errors=on_errors)
        self.mol = AtomPairFP(n_jobs, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        self.mol = TopologicalTorsionFP(n_jobs, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        self.mol = MACCS(n_jobs, input_type=input_type, on_errors=on_errors)
        self.mol = ECFP(n_jobs, radius=radius, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        self.mol = FCFP(n_jobs, radius=radius, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        self.mol = DescriptorFeature(n_jobs, input_type=input_type, on_errors=on_errors)
