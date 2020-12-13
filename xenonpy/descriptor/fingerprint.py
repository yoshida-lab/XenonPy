#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
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

    def __init__(self, n_jobs=-1,
                 *, input_type='mol', on_errors='raise', return_type='any', target_col=None):
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
        """
        # self.arg = arg # arg[0] = radius, arg[1] = bit length
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self.input_type = input_type
        nms = [x[0] for x in Descriptors._descList]
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)
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
        return [x[0] for x in Descriptors._descList]
        # return ['desc200:' + str(i) for i in range(200)]


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
