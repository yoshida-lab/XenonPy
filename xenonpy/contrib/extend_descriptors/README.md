# Extend Descriptors

## FrozenFeaturizerDescriptor

This is a sample code for creating artificial descriptor based on a trained neural network.
This code creates a BaseFeaturizer object in XenonPy that can be used as input for training models.
The input is in the same format as the input of the descriptor used in the neural network.

By passing both the XenonPy descriptor object and XenonPy frozen featurizer object into this class when creating the Base Featurizer, the output will be a dataframe same as other typical XenonPy descriptors, while the number of columns is the number of neurons in the chosen hidden layers.


## Mordred2DDescriptor

This is a sample code for calculating the 2D Mordred descriptor:
https://github.com/mordred-descriptor/mordred

This code creates a BaseFeaturizer object in XenonPy that can be used as input for training models.

## OrganicCompDescriptor

This is a sample code for calculating the XenonPy compositional descriptors for organic molecules in SMILES or RDKit MOL format:
https://xenonpy.readthedocs.io/en/latest/features.html#compositional-descriptors

This code creates a BaseFeaturizer object in XenonPy that can be used as input for training models.

-----------
written by Stephen Wu, 2019.05.31
updated by Stephen Wu, 2019.07.11
