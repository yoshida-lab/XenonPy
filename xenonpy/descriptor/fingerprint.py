from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import MACCSkeys as MAC
from rdkit.Chem import rdMolDescriptors as rdMol
from rdkit.ML.Descriptors import MoleculeDescriptors

from .base import BaseDescriptor, BaseFeaturizer


class RDKitFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, fp_size=2048):
        """
        Base class for composition feature.
        """

        self.fp_size = fp_size

    def featurize(self, x):
        return list(Chem.RDKFingerprint(x, fpSize=self.fp_size))

    @property
    def feature_labels(self):
        return ["rdkit:" + str(i) for i in range(self.fp_size)]


class AtomPairFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048):
        """
        Atom Pair fingerprints
            Returns the atom-pair fingerprint for a molecule.The algorithm used is described here: 
            R.E. Carhart, D.H. Smith, R. Venkataraghavan; 
            "Atom Pairs as Molecular Features in Structure-Activity Studies: Definition and Applications" 
            JCICS 25, 64-73 (1985).
            This is currently just in binary bits with fixed length after folding.

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


class TopologicalTorsionFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, n_bits=2048):
        """
        Topological Torsion fingerprints
            Returns the topological-torsion fingerprint for a molecule.
            This is currently just in binary bits with fixed length after folding.

        Parameters
        ----------
        n_bits: int
           Fixed bit length based on folding.

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
        The MACCS keys for a molecule. The result is a 167-bit vector. There are 166 public keys, but to maintain consistency with other software packages they are numbered from 1.
        """
        super().__init__(n_jobs=n_jobs)

    def featurize(self, x):
        return list(MAC.GenMACCSKeys(x))

    @property
    def feature_labels(self):
        return ['maccs:' + str(i) for i in range(167)]


class FCFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048):
        """
        Morgan (Circular) fingerprints + feature-based (FCFP)
            The algorithm used is described in the paper Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. JCIM 50:742-54 (2010)

        Parameters
        ----------
        radius: int
            The radius parameter in the Morgan fingerprints, which is roughly half of the diameter parameter in FCFP, i.e., radius=2 is roughly equivalent to FCFP4.
        n_bits: int
            Fixed bit length based on folding.
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


class ECFP(BaseFeaturizer):

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048):
        """
        Morgan (Circular) fingerprints (ECFP)
            The algorithm used is described in the paper Rogers, D. & Hahn, M. Extended-Connectivity Fingerprints. JCIM 50:742-54 (2010)

        Parameters
        ----------
        radius: int
            The radius parameter in the Morgan fingerprints, which is roughly half of the diameter parameter in ECFP, i.e., radius=2 is roughly equivalent to ECFP4.
        n_bits: int
            Fixed bit length based on folding.
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


class DescriptorFeature(BaseFeaturizer):

    def __init__(self, n_jobs=-1):
        """
        All descriptors in RDKit (length = 200) [may include NaN]
            see https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors for the full list
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
    Calculate fingerprints or descriptors of organic molecules.
    """

    def __init__(self, n_jobs=-1, *, radius=3, n_bits=2048, fp_size=2048):
        """

        Parameters
        ----------
        radius: int
            The radius parameter in the Morgan fingerprints, which is roughly half of the diameter parameter in ECFP/FCFP, i.e., radius=2 is roughly equivalent to ECFP4/FCFP4.
        n_bits: int
            Fixed bit length based on folding.
        useFeatures: bool
        """

        super().__init__()
        self.n_jobs = n_jobs

        self.mol = RDKitFP(n_jobs, fp_size=fp_size)
        self.mol = AtomPairFP(n_jobs, n_bits=n_bits)
        self.mol = TopologicalTorsionFP(n_jobs, n_bits=n_bits)
        self.mol = MACCS(n_jobs)
        self.mol = ECFP(n_jobs, radius=radius, n_bits=n_bits)
        self.mol = FCFP(n_jobs, radius=radius, n_bits=n_bits)
        self.rdkit_desc = DescriptorFeature(n_jobs)
