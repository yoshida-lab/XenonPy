from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys as MAC
from rdkit.Chem import rdMolDescriptors as rdMol
from rdkit.ML.Descriptors import MoleculeDescriptors

from .base import BaseDescriptor, BaseFeaturizer


class AtomPairFingerprint(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048):
        """
        Atom Pair fingerprints

        Parameters
        ----------
        n_bits: int
           Fixed bit length based on folding.
        """
        super().__init__(n_jobs=n_jobs)
        self.n_bits = n_bits

    def featurize(self, x):
        return list(rdMol.GetHashedAtomPairFingerprintAsBitVect(x, n_bits=self.n_bits))

    @property
    def feature_labels(self):
        return ['apfp:' + str(i) for i in range(self.n_bits)]


class TopologicalTorsionFingerprint(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048):
        """
        Topological Torsion fingerprints

        Parameters
        ----------
        n_bits: int
           bit length

        """
        super().__init__(n_jobs=n_jobs)
        self.n_bits = n_bits

    def featurize(self, x):
        return list(rdMol.GetHashedTopologicalTorsionFingerprintAsBitVect(x, n_bits=self.n_bits))

    @property
    def feature_labels(self):
        return ['ttfp:' + str(i) for i in range(self.n_bits)]


class MACCS(BaseFeaturizer):

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


class MorganFingerprintiWithFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048):
        """
        Morgan (Circular) fingerprints + feature-based (FCFP)

        Parameters
        ----------
        radius: int
        n_bits: int
           bit length.
        useFeatures: bool
        """
        super().__init__(n_jobs=n_jobs)
        self.radius = radius
        self.n_bits = n_bits
        # self.arg = arg # arg[0] = radius, arg[1] = bit length

    def featurize(self, x):
        return list(
            rdMol.GetMorganFingerprintAsBitVect(
                x, self.radius, n_bits=self.n_bits, useFeatures=True))

    @property
    def feature_labels(self):
        return ['fcfp3:' + str(i) for i in range(self.n_bits)]


class MorganFingerprint(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048):
        """
        Morgan (Circular) fingerprints (ECFP)

        Parameters
        ----------
        n_bits: int
           bit length.
        """
        super().__init__(n_jobs=n_jobs)
        self.radius = radius
        self.n_bits = n_bits
        # self.arg = arg # arg[0] = radius, arg[1] = bit length

    def featurize(self, x):
        return list(rdMol.GetMorganFingerprintAsBitVect(x, self.radius, n_bits=self.n_bits))

    @property
    def feature_labels(self):
        return ['ecfp3:' + str(i) for i in range(self.n_bits)]


class MolecularDescriptor(BaseFeaturizer):

    def __init__(self, n_jobs=-1):
        """
        All descriptors in R (length = 200) [Question: maybe include NaN]

        """
        # self.arg = arg # arg[0] = radius, arg[1] = bit length
        super().__init__(n_jobs=n_jobs)
        nms = [x[0] for x in Descriptors._descList]
        self.calc = MoleculeDescriptors.MolecularDescriptorCalculator(nms)

    def featurize(self, x):
        return self.calc.CalcDescriptors(x)

    @property
    def feature_labels(self):
        return ['desc200:' + str(i) for i in range(200)]


class Fingerprints(BaseDescriptor):
    """
    Calculate elemental descriptors from compound's composition.
    """

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048):
        """

        Parameters
        ----------
        radius: int
        n_bits: int
           bit length.
        useFeatures: bool
        """

        super().__init__()
        self.n_jobs = n_jobs

        self.mol = AtomPairFingerprint(n_jobs, n_bits=n_bits)
        self.mol = TopologicalTorsionFingerprint(n_jobs, n_bits=n_bits)
        self.mol = MACCS(n_jobs)
        self.mol = MorganFingerprint(n_jobs, radius=radius, n_bits=n_bits)
        self.mol = MorganFingerprint(n_jobs, radius=radius, n_bits=n_bits)
        # self.rdkit_desc = Desc200Feature(n_jobs)
