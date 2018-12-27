import numpy as np
import pandas as pd
from rdkit import Chem

from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys as MAC
from rdkit.Chem import rdMolDescriptors as rdMol
from rdkit.ML.Descriptors import MoleculeDescriptors

from .base import BaseDescriptor, BaseFeaturizer


class APFPFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, nBits=2048):
        """
        Atom Pair fingerprints

        Parameters
        ----------
        nBits: int
           Fixed bit length based on folding.
        """
        super().__init__(n_jobs=n_jobs)
        self.nBits = nBits

    def featurize(self, x):
        return list(rdMol.GetHashedAtomPairFingerprintAsBitVect(x, nBits=self.nBits))

    @property
    def feature_labels(self):
        return ['apfp:' + str(i) for i in range(self.nBits)]


class TTFPFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, nBits=2048):
        """
        Topological Torsion fingerprints

        Parameters
        ----------
        nBits: int
           bit length

        """
        super().__init__(n_jobs=n_jobs)
        self.nBits = nBits

    def featurize(self, x):
        return list(rdMol.GetHashedTopologicalTorsionFingerprintAsBitVect(x, nBits=self.nBits))

    @property
    def feature_labels(self):
        return ['ttfp:' + str(i) for i in range(self.nBits)]


class MACCSFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1):
        """
        MACCS Keys, length fixed at 167
        """
        super().__init__(n_jobs=n_jobs)

    def featurize(self, x):
        return list(MAC.GenMACCSKeys(x))

    @property
    def feature_labels(self):
        return ['maccs:' + str(i) for i in range(167)]


class FCFP3Feature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, nBits=2048, useFeatures=True):
        """
        Morgan (Circular) fingerprints + feature-based (FCFP)

        Parameters
        ----------
        radius: int
        nBits: int
           bit length.
        useFeatures: bool
        """
        super().__init__(n_jobs=n_jobs)
        self.radius = radius
        self.nBits = nBits
        self.useFeatures = useFeatures
        #self.arg = arg # arg[0] = radius, arg[1] = bit length

    def featurize(self, x):
        return list(
            rdMol.GetMorganFingerprintAsBitVect(
                x, self.radius, nBits=self.nBits, useFeatures=self.useFeatures))

    @property
    def feature_labels(self):
        return ['fcfp3:' + str(i) for i in range(self.nBits)]


class ECFP3Feature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, nBits=2048):
        """
        Morgan (Circular) fingerprints (ECFP)

        Parameters
        ----------
        nBits: int
           bit length.
        """
        super().__init__(n_jobs=n_jobs)
        self.radius = radius
        self.nBits = nBits
        #self.arg = arg # arg[0] = radius, arg[1] = bit length

    def featurize(self, x):
        return list(rdMol.GetMorganFingerprintAsBitVect(x, self.radius, nBits=self.nBits))

    @property
    def feature_labels(self):
        return ['ecfp3:' + str(i) for i in range(self.nBits)]


class Desc200Feature(BaseFeaturizer):

    def __init__(self, n_jobs=-1):
        """
        All descriptors in R (length = 200) [Question: maybe include NaN]

        """
        #self.arg = arg # arg[0] = radius, arg[1] = bit length
        super().__init__(n_jobs=n_jobs)
        nms = [x[0] for x in Descriptors._descList]
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

    def featurize(self, x):
        return self.calc.CalcDescriptors(x)

    @property
    def feature_labels(self):
        return ['desc200:' + str(i) for i in range(200)]


class RdkitFingerprint(BaseDescriptor):
    """
    Calculate elemental descriptors from compound's composition.
    """

    def __init__(self, n_jobs=-1, *, radius=3, nBits=2048, useFeatures=True):
        """

        Parameters
        ----------
        radius: int
        nBits: int
           bit length.
        useFeatures: bool
        """

        super().__init__()
        self.n_jobs = n_jobs

        self.rdkit_fp = APFPFeature(n_jobs, nBits=nBits)
        self.rdkit_fp = TTFPFeature(n_jobs, nBits=nBits)
        self.rdkit_fp = MACCSFeature(n_jobs)
        self.rdkit_fp = FCFP3Feature(n_jobs, radius=radius, nBits=nBits, useFeatures=useFeatures)
        self.rdkit_fp = ECFP3Feature(n_jobs, radius=radius, nBits=nBits)
        #self.rdkit_desc = Desc200Feature(n_jobs)
