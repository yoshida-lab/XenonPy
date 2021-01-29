from typing import Union
from xenonpy.descriptor.base import BaseFeaturizer, BaseDescriptor

from xenonpy.model.training import Trainer
from xenonpy.datatools import Scaler
import torch, numpy


class NNPropDescriptor(BaseFeaturizer):

    def __init__(self, descriptor_calculator: Union[BaseDescriptor, BaseFeaturizer],
                 nn_trainer: Trainer,
                 scaler: Scaler = None, *,
                 on_errors='nan',
                 return_type='any'):
        """
        A featurizer for extracting artificial descriptors from neural networks
        Parameters
        ----------
        descriptor_calculator : BaseFeaturizer or BaseDescriptor
            Convert input data into descriptors to keep consistency with the pre-trained model.
        nn_trainer : Trainer
            XenonPy Trainer that contains the neural network set to the right checkpoint
        scaler : Scaler
            XenonPy Scaler for the output
        """

        # fix n_jobs to be 0 to skip automatic wrapper in XenonPy BaseFeaturizer class
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)
        self.FP = descriptor_calculator
        self.FP.on_errors = on_errors
        self.FP.return_type = return_type
        self.nn = nn_trainer
        self.scaler = scaler
        self.colnames = ['thermal_conductivity', 'thermal_diffusivity', 'density',
                         'static_dielectric_const', 'nematic_order_param', 'Cp', 'Cv',
                         'compress_T', 'compress_S', 'bulk_modulus_T', 'bulk_modulus_S',
                         'speed_of_sound', 'volume_expansion', 'linear_expansion']
        self.__authors__ = ['Stephen Wu']

    def featurize(self, x, *, ori_scale=False):  # scalingするかどうかを選択
        # transform input to descriptor dataframe
        tmp_df = self.FP.transform(x)
        # convert descriptor dataframe to hidden layer dataframe
        output = self.nn.predict(x_in=torch.tensor(tmp_df.values, dtype=torch.float)).detach().numpy()
        if ori_scale and self.scaler is not None:
            return pd.DataFrame(self.scaler.inverse_transform(output), index=tmp_df.index, columns=self.colnames)
        else:
            return pd.DataFrame(output, index=tmp_df.index, columns=self.colnames)

    @property
    def feature_labels(self):
        return self.colnames
#branch2
