# XenonPy BaseFeaturizer for extracting artificial descriptors from neural networks
# Input:
# (1) xenonpy frozen featurizer object extracting hidden layers in a neural network
# (2) xenonpy descriptor object used for the input of the neural network

from xenonpy.descriptor.base import BaseFeaturizer

class FrozenFeaturizerDescriptor(BaseFeaturizer):

    def __init__(self, Fingerprint_obj, FrozenFeaturizer_obj, *, on_errors='raise', return_type='any'):
        # fix n_jobs to be 0 to skip automatic wrapper in XenonPy BaseFeaturizer class
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)
        self.FP = Fingerprint_obj
        self.ff = FrozenFeaturizer_obj

    def featurize(self, x):
        # transform input to descriptor dataframe
        tmp_df = self.FP.transform(x)
        # convert descriptor dataframe to hidden layer dataframe
        self.output = self.ff.transform(tmp_df, depth=1 ,return_type='df')
        return self.output
    
    @property
    def feature_labels(self):
        # column names based on xenonpy frozen featurizer setting
        return self.output.columns
