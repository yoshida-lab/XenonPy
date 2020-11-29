#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from typing import Union

from xenonpy.descriptor.base import BaseFeaturizer, BaseDescriptor


class ReactionDescriptor(BaseFeaturizer):

    def __init__(self,
                 descriptor_calculator: Union[BaseDescriptor, BaseFeaturizer],
                 *,
                 on_errors='raise',
                 return_type='any',
                 target_col=None):
        """
        A featurizer wrapper to handle dataframe input
        Parameters
        ----------
        descriptor_calculator : BaseFeaturizer or BaseDescriptor
            Convert input data into selected descriptors.
        """

        # fix n_jobs to be 0 to skip automatic wrapper in XenonPy BaseFeaturizer class
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)
        self.FP = descriptor_calculator
        self.FP.on_errors = on_errors
        self.FP.return_type = return_type
        self.target_col = target_col
        self.output = None
        self.__authors__ = ['Stephen Wu', 'TsumiNa']

    def featurize(self, x):
        target_col = self.target_col
        if target_col is None:
            target_col = x.columns[0]
        self.output = self.FP.transform(x[target_col])
        return self.output

    @property
    def feature_labels(self):
        # column names based on xenonpy frozen featurizer setting
        return self.output.columns
