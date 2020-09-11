#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union
from rdkit import Chem

from xenonpy.descriptor.base import BaseFeaturizer, BaseDescriptor
from xenonpy.contrib.ismd import ReactantPool
import numpy as np
import pandas as pd


class ReactionDescriptor(BaseFeaturizer):

    def __init__(self,
                 descriptor_calculator: Union[BaseDescriptor, BaseFeaturizer],
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

    def product_validation(self, product):
        if Chem.MolFromSmiles(product) is not None:
            return True
        else:
            return False

    def featurize(self, samples):
        # reacte input reactants to product
        samples["reactant_SMILES"] = self.pool.index2reactant(samples)
        _, samples["product"] = self.reactor.react(samples["reactant_SMILES"])
        samples["validate"] = list(map(self.product_validation, samples["product"]))
        self.df = samples
        valid_product = samples["product"].loc[samples["validate"] == True]
        # transform input to descriptor dataframe
        product_FP = self.FP.transform(valid_product)
        # print(x_df.loc[x_df["validate"]==False])
        return product_FP

    @property
    def feature_labels(self):
        return None
