from rdkit import Chem
from xenonpy.descriptor import *
from xenonpy.descriptor.base import BaseDescriptor, BaseFeaturizer
import random


class R2Fingerprints(BaseDescriptor):
    """
    Calculate fingerprints or descriptors of organic molecules.
    """

    def __init__(self,
                 reactor,
                 n_jobs=-1,
                 *,
                 radius=3,
                 n_bits=2048,
                 fp_size=2048,
                 input_type='mol',
                 featurizers='all',
                 on_errors='raise'):
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
        self.reactor = reactor

        self.mol = RDKitFP(n_jobs, fp_size=fp_size,
                           input_type=input_type, on_errors=on_errors)
        self.mol = AtomPairFP(n_jobs, n_bits=n_bits,
                              input_type=input_type, on_errors=on_errors)
        self.mol = TopologicalTorsionFP(
            n_jobs, n_bits=n_bits, input_type=input_type, on_errors=on_errors)
        self.mol = MACCS(n_jobs, input_type=input_type, on_errors=on_errors)
        self.mol = ECFP(n_jobs, radius=radius, n_bits=n_bits,
                        input_type=input_type, on_errors=on_errors)
        self.mol = FCFP(n_jobs, radius=radius, n_bits=n_bits,
                        input_type=input_type, on_errors=on_errors)
        self.mol = DescriptorFeature(
            n_jobs, input_type=input_type, on_errors=on_errors)

    def reactant_transform(self, X, **kwargs):
        _, product = self.reactor.react(X)
        valid_product = [
            p for p in product if Chem.MolFromSmiles(p) is not None]
        n_substitute = len(X) - len(valid_product)
        substitute = random.choices(valid_product, k=n_substitute)
        valid_product = valid_product + substitute
        result = super().transform(valid_product, **kwargs)

        return result
