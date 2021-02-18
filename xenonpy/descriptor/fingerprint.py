#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors as ChemDesc
from rdkit.Chem import MACCSkeys as MAC
from rdkit.Chem import rdMolDescriptors as rdMol
from rdkit.Chem import rdmolops as rdm
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from rdkit.ML.Descriptors import MoleculeDescriptors

from scipy.sparse import coo_matrix

from xenonpy.descriptor.base import BaseDescriptor, BaseFeaturizer

__all__ = ['RDKitFP', 'AtomPairFP', 'TopologicalTorsionFP', 'MACCS', 'FCFP', 'ECFP', 'PatternFP', 'LayeredFP',
           'MHFP', 'DescriptorFeature', 'Fingerprints']


def count_fp(fp, dim=2**10):
    tmp = fp.GetNonzeroElements()
    return coo_matrix((list(tmp.values()), (np.repeat(0, len(tmp)), [i % dim for i in tmp.keys()])),
                      shape=(1, dim)).toarray().flatten()


class RDKitFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048, bit_per_entry=None, counting=False,
                 input_type='mol', on_errors='raise', return_type='any', target_col=None):
        """
        RDKit fingerprint.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
        n_bits: int
            Fingerprint size.
        bit_per_entry: int
            Number of bits used to represent a single entry (only for non-counting case).
            Default value follows rdkit default.
        counting: boolean
            Record counts of the entries instead of bits only.
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.n_bits = n_bits
        if bit_per_entry is None:
            self.bit_per_entry = 2
        else:
            self.bit_per_entry = bit_per_entry
        self.counting = counting
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)

        if self.counting:
            return count_fp(rdm.UnfoldedRDKFingerprintCountBased(x), dim=self.n_bits)
        else:
            return list(Chem.RDKFingerprint(x, fpSize=self.n_bits, nBitsPerHash=self.bit_per_entry))

    @property
    def feature_labels(self):
        if self.counting:
            return ["rdkit_c:" + str(i) for i in range(self.n_bits)]
        else:
            return ["rdkit:" + str(i) for i in range(self.n_bits)]


class AtomPairFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048, bit_per_entry=None, counting=False,
                 input_type='mol', on_errors='raise', return_type='any', target_col=None):
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
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
        n_bits: int
           Fixed bit length based on folding.
        bit_per_entry: int
            Number of bits used to represent a single entry (only for non-counting case).
            Default value follows rdkit default.
        counting: boolean
            Record counts of the entries instead of bits only.
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.n_bits = n_bits
        if bit_per_entry is None:
            self.bit_per_entry = 4
        else:
            self.bit_per_entry = bit_per_entry
        self.counting = counting
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.counting:
            return count_fp(rdMol.GetHashedAtomPairFingerprint(x, nBits=self.n_bits), dim=self.n_bits)
        else:
            return list(rdMol.GetHashedAtomPairFingerprintAsBitVect(x, nBits=self.n_bits,
                                                                    nBitsPerEntry=self.bit_per_entry))

    @property
    def feature_labels(self):
        if self.counting:
            return ['apfp_c:' + str(i) for i in range(self.n_bits)]
        else:
            return ['apfp:' + str(i) for i in range(self.n_bits)]


class TopologicalTorsionFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048, bit_per_entry=None, counting=False,
                 input_type='mol', on_errors='raise', return_type='any', target_col=None):
        """
        Topological Torsion fingerprints.
        Returns the topological-torsion fingerprint for a molecule.
        This is currently just in binary bits with fixed length after folding.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
        n_bits: int
           Fixed bit length based on folding.
        bit_per_entry: int
            Number of bits used to represent a single entry (only for non-counting case).
            Default value follows rdkit default.
        counting: boolean
            Record counts of the entries instead of bits only.
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.n_bits = n_bits
        if bit_per_entry is None:
            self.bit_per_entry = 4
        else:
            self.bit_per_entry = bit_per_entry
        self.counting = counting
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.counting:
            return count_fp(rdMol.GetHashedTopologicalTorsionFingerprint(x, nBits=self.n_bits), dim=self.n_bits)
        else:
            return list(rdMol.GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits=self.n_bits,
                                                                              nBitsPerEntry=self.bit_per_entry))

    @property
    def feature_labels(self):
        if self.counting:
            return ['ttfp_c:' + str(i) for i in range(self.n_bits)]
        else:
            return ['ttfp:' + str(i) for i in range(self.n_bits)]


class MACCS(BaseFeaturizer):

    def __init__(self, n_jobs=-1,
                 *, input_type='mol', on_errors='raise', return_type='any', target_col=None):
        """
        The MACCS keys for a molecule. The result is a 167-bit vector. There are 166 public keys,
        but to maintain consistency with other software packages they are numbered from 1.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        return list(MAC.GenMACCSKeys(x))

    @property
    def feature_labels(self):
        return ['maccs:' + str(i) for i in range(167)]


class FCFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048, counting=False,
                 input_type='mol', on_errors='raise', return_type='any', target_col=None):
        """
        Morgan (Circular) fingerprints + feature-based (FCFP)
        The algorithm used is described in the paper Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints.
        JCIM 50:742-54 (2010)

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
        radius: int
            The radius parameter in the Morgan fingerprints, which is roughly half of the diameter parameter in FCFP,
            i.e., radius=2 is roughly equivalent to FCFP4.
        n_bits: int
            Fixed bit length based on folding.
        counting: boolean
            Record counts of the entries instead of bits only.
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.radius = radius
        self.n_bits = n_bits
        self.counting = counting
        self.__authors__ = ['Stephen Wu', 'TsumiNa']
        # self.arg = arg # arg[0] = radius, arg[1] = bit length

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.counting:
            return count_fp(rdMol.GetHashedMorganFingerprint(
                x, radius=self.radius, nBits=self.n_bits, useFeatures=True), dim=self.n_bits)
        else:
            return list(rdMol.GetMorganFingerprintAsBitVect(
                x, radius=self.radius, nBits=self.n_bits, useFeatures=True))

    @property
    def feature_labels(self):
        if self.counting:
            return [f'fcfp{self.radius * 2}_c:' + str(i) for i in range(self.n_bits)]
        else:
            return [f'fcfp{self.radius * 2}:' + str(i) for i in range(self.n_bits)]


class ECFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048, counting=False,
                 input_type='mol', on_errors='raise', return_type='any', target_col=None):
        """
        Morgan (Circular) fingerprints (ECFP)
        The algorithm used is described in the paper Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints.
        JCIM 50:742-54 (2010)

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
        radius: int
            The radius parameter in the Morgan fingerprints, which is roughly half of the diameter parameter in ECFP,
            i.e., radius=2 is roughly equivalent to ECFP4.
        n_bits: int
            Fixed bit length based on folding.
        counting: boolean
            Record counts of the entries instead of bits only.
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.radius = radius
        self.n_bits = n_bits
        self.counting = counting
        self.__authors__ = ['Stephen Wu', 'TsumiNa']
        # self.arg = arg # arg[0] = radius, arg[1] = bit length

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.counting:
            return count_fp(rdMol.GetHashedMorganFingerprint(x, radius=self.radius,
                                                             nBits=self.n_bits), dim=self.n_bits)
        else:
            return list(rdMol.GetMorganFingerprintAsBitVect(x, radius=self.radius, nBits=self.n_bits))

    @property
    def feature_labels(self):
        if self.counting:
            return [f'ecfp{self.radius * 2}_c:' + str(i) for i in range(self.n_bits)]
        else:
            return [f'ecfp{self.radius * 2}:' + str(i) for i in range(self.n_bits)]


class PatternFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048,
                 input_type='mol', on_errors='raise', return_type='any', target_col=None):
        """
        A fingerprint designed to be used in substructure screening using SMARTS patterns (unique in RDKit).

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.n_bits = n_bits
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        return list(rdm.PatternFingerprint(x, fpSize=self.n_bits))

    @property
    def feature_labels(self):
        return ['patfp:' + str(i) for i in range(self.n_bits)]


class LayeredFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048,
                 input_type='mol', on_errors='raise', return_type='any', target_col=None):
        """
        A substructure fingerprint that is more complex than PatternFP (unique in RDKit).

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.n_bits = n_bits
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        return list(rdm.LayeredFingerprint(x, fpSize=self.n_bits))

    @property
    def feature_labels(self):
        return ['layfp:' + str(i) for i in range(self.n_bits)]


class MHFP(BaseFeaturizer):

    def __init__(self, n_jobs=1, *, radius=3, n_bits=2048,
                 input_type='mol', on_errors='raise', return_type='any', target_col=None):
        """
        Variation from the MinHash fingerprint, which is based on ECFP with
        locality sensitive hashing to increase compactness of information during hashing.
        The algorithm used is described in the paper
        Probst, D. & Reymond, J.-L., A probabilistic molecular fingerprint for big data settings.
        Journal of Cheminformatics, 10:66 (2018)

        Note that MHFP currently does not support parallel computing, so please fix n_jobs to 1.

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
        radius: int
            The radius parameter in the SECFP(RDKit version) fingerprints,
            which is roughly half of the diameter parameter in ECFP,
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        self.radius = radius
        self.n_bits = n_bits
        self.mhfp = MHFPEncoder()
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        return list(self.mhfp.EncodeSECFPMol(x, radius=self.radius, length=self.n_bits))

    @property
    def feature_labels(self):
        return [f'secfp{self.radius * 2}:' + str(i) for i in range(self.n_bits)]


class DescriptorFeature(BaseFeaturizer):

    classic = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex', 'qed', 'MolWt',
               'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons', 'MaxPartialCharge',
               'MinPartialCharge', 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2',
               'FpDensityMorgan3', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n',
               'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3',
               'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14',
               'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9',
               'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7',
               'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2',
               'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA',
               'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4',
               'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'VSA_EState1', 'VSA_EState10',
               'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
               'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
               'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
               'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
               'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP',
               'MolMR', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH',
               'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
               'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH',
               'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',
               'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
               'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan',
               'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan',
               'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy',
               'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso',
               'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid',
               'fr_phos_ester', 'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine',
               'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
               'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

    def __init__(self, n_jobs=-1,
                 *, input_type='mol', on_errors='raise', return_type='any', target_col=None, desc_list='all'):
        """
        All descriptors in RDKit (length = 200) [may include NaN]
            see https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors for the full list

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cups. Set -1 to use all cpu cores (default).
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        desc_list: string or list
            List of descriptor names to be called in rdkit to calculate molecule descriptors.
            If ``classic``, the full list of rdkit v.2020.03.xx is used. (length = 200)
            Default is to use the latest list available in the rdkit. (length = 208 in rdkit v.2020.09.xx)
        """
        # self.arg = arg # arg[0] = radius, arg[1] = bit length
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        if desc_list == 'all':
            self.nms = [x[0] for x in ChemDesc._descList]
        elif desc_list == 'classic':
            self.nms = self.classic
        else:
            self.nms = desc_list
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(self.nms)
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        if self.input_type == 'smiles':
            x_ = x
            x = Chem.MolFromSmiles(x)
            if x is None:
                raise ValueError('cannot convert Mol from SMILES %s' % x_)
        if self.input_type == 'any':
            if not isinstance(x, Chem.rdchem.Mol):
                x_ = x
                x = Chem.MolFromSmiles(x)
                if x is None:
                    raise ValueError('cannot convert Mol from SMILES %s' % x_)
        return self.calc.CalcDescriptors(x)

    @property
    def feature_labels(self):
        return self.nms


class Fingerprints(BaseDescriptor):
    """
    Calculate fingerprints or descriptors of organic molecules.
    Note that MHFP currently does not support parallel computing, so n_jobs is fixed to 1.
    """

    def __init__(self,
                 n_jobs=-1,
                 *,
                 radius=3,
                 n_bits=2048,
                 bit_per_entry=None,
                 counting=False,
                 input_type='mol',
                 featurizers='all',
                 on_errors='raise',
                 target_col=None):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Can be -1 or # of cpus. Set -1 to use all cpu cores (default).
        radius: int
            The radius parameter in the Morgan fingerprints,
            which is roughly half of the diameter parameter in ECFP/FCFP,
            i.e., radius=2 is roughly equivalent to ECFP4/FCFP4.
        n_bits: int
            Fixed bit length based on folding.
        bit_per_entry: int
            Number of bits used to represent a single entry (only for non-counting case)
            in RDKitFP, AtomPairFP, and TopologicalTorsionFP.
            Default value follows rdkit default.
        counting: boolean
            Record counts of the entries instead of bits only.
        featurizers: list[str] or str or 'all'
            Featurizer(s) that will be used.
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """

        super().__init__(featurizers=featurizers)

        self.mol = RDKitFP(n_jobs, n_bits=n_bits, bit_per_entry=bit_per_entry, counting=counting,
                           input_type=input_type, on_errors=on_errors, target_col=target_col)
        self.mol = AtomPairFP(n_jobs, n_bits=n_bits, bit_per_entry=bit_per_entry, counting=counting,
                              input_type=input_type, on_errors=on_errors, target_col=target_col)
        self.mol = TopologicalTorsionFP(n_jobs, n_bits=n_bits, input_type=input_type, bit_per_entry=bit_per_entry,
                                        counting=counting, on_errors=on_errors, target_col=target_col)
        self.mol = MACCS(n_jobs, input_type=input_type, on_errors=on_errors, target_col=target_col)
        self.mol = ECFP(n_jobs, radius=radius, n_bits=n_bits, input_type=input_type, counting=counting,
                        on_errors=on_errors, target_col=target_col)
        self.mol = FCFP(n_jobs, radius=radius, n_bits=n_bits, input_type=input_type, counting=counting,
                        on_errors=on_errors, target_col=target_col)
        self.mol = PatternFP(n_jobs, n_bits=n_bits, input_type=input_type, on_errors=on_errors, target_col=target_col)
        self.mol = LayeredFP(n_jobs, n_bits=n_bits, input_type=input_type, on_errors=on_errors, target_col=target_col)
        #         self.mol = SECFP(n_jobs, radius=radius, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        self.mol = MHFP(1, radius=radius, n_bits=n_bits,
                        input_type=input_type, on_errors=on_errors, target_col=target_col)
        self.mol = DescriptorFeature(n_jobs, input_type=input_type,
                                     on_errors=on_errors, target_col=target_col)
