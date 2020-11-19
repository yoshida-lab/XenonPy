#  Copyright (c) 2019. stewu5. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union

from xenonpy.descriptor import FrozenFeaturizer
from xenonpy.descriptor.base import BaseFeaturizer, BaseDescriptor


class FrozenFeaturizerDescriptor(BaseFeaturizer):

    def __init__(self, descriptor_calculator: Union[BaseDescriptor, BaseFeaturizer],
                 frozen_featurizer: FrozenFeaturizer, *,
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
        self.FP.on_errors = on_errors
        self.FP.return_type = return_type
        self.ff = frozen_featurizer
        self.output = None
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x, *, depth=1):
        # transform input to descriptor dataframe
        tmp_df = self.FP.transform(x)
        # convert descriptor dataframe to hidden layer dataframe
        self.output = self.ff.transform(tmp_df, depth=depth, return_type='df')
        return self.output

    @property
    def feature_labels(self):
        # column names based on xenonpy frozen featurizer setting
        return self.output.columns
