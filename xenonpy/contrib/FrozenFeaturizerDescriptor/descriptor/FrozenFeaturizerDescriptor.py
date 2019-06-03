# XenonPy BaseFeaturizer for extracting artificial descriptors from neural networks
# Input:
# (1) xenonpy frozen featurizer object extracting hidden layers in a neural network
# (2) xenonpy descriptor object used for the input of the neural network

from typing import Union

from xenonpy.descriptor import FrozenFeaturizer
from xenonpy.descriptor.base import BaseFeaturizer, BaseDescriptor


class FrozenFeaturizerDescriptor(BaseFeaturizer):

    def __init__(self, fingerprint: Union[BaseDescriptor, BaseFeaturizer], frozen_featurizer: FrozenFeaturizer, *,
                 on_errors='raise',
                 return_type='any'):
        """

        Parameters
        ----------
        fingerprint : BaseFeaturizer or BaseDescriptor
        frozen_featurizer : FrozenFeaturizer
        """
        # fix n_jobs to be 0 to skip automatic wrapper in XenonPy BaseFeaturizer class
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)
        self.FP = fingerprint
        self.ff = frozen_featurizer
        self.output = None

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
