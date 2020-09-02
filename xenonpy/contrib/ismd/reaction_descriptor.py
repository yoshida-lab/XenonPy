from typing import Union
from rdkit import Chem

from xenonpy.descriptor.base import BaseFeaturizer, BaseDescriptor
from xenonpy.contrib.ismd import ReactantPool
import numpy as np


class ReactionDescriptor(BaseFeaturizer):

    def __init__(self, descriptor_calculator: Union[BaseDescriptor, BaseFeaturizer],
                 reactor,
                 reactant_pool: ReactantPool,
                 on_errors='raise',
                 return_type='any'):
        """
        A featurizer for extracting artificial descriptors from neural networks
        Parameters
        ----------
        descriptor_calculator : BaseFeaturizer or BaseDescriptor
            Convert input data into descriptors to keep consistency with the pre-trained model.
        frozen_featurizer : FrozenFeaturizer
            Extracting artificial descriptors from neural networks
        """

        # fix n_jobs to be 0 to skip automatic wrapper in XenonPy BaseFeaturizer class
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)
        self.FP = descriptor_calculator
        self.reactor = reactor
        self.pool = reactant_pool
        self.output = None

    def featurize(self, x):
        # reacte input reactants to product
        reactant = self.pool.index2reactant(x)
        _, product = self.reactor.react(reactant)
        valid_product = [p for p in product if Chem.MolFromSmiles(p) is not None]
        n_substitute = len(x) - len(valid_product)
        substitute = np.random.choice(a=valid_product, size=n_substitute)
        valid_product = np.concatenate((valid_product,substitute),axis=0)
        # transform input to descriptor dataframe
        self.output = self.FP.transform(valid_product)

        return self.output

    @property
    def feature_labels(self):
        # column names based on xenonpy frozen featurizer setting
        return self.output.columns
