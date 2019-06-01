# FrozenFeaturizerDescriptor

This is a sample code for creating artificial descriptor based on a trained neural network.
This code creates a BaseFeaturizer object in XenonPy that can be used as input for training models.
The input is in the same format as the input of the descriptor used in the neural network.

By passing both the XenonPy descriptor object and XenonPy frozen featurizer object into this class when creating the Base Featurizer, the output will be a dataframe same as other typical XenonPy descriptors, while the number of columns is the number of neurons in the chosen hidden layers.

-----------
written by Stephen Wu, 2019.05.31
