# New BaseProposal class

## reactant_pool

This is a sample code of a new class of BaseProposal, called ReactantPool.
ReactantPool allows generation of molecules based on virtual reaction of reactants in a predefined pool of reactants. Initialization of this class requires a pretrained reactor, a ``pandas.DataFrame`` of reactant candidates, and a pre-calculated similarity matrix between all pairs of reactant candidates. Based on an initial set of reactant pairs, new reactant pairs are randomly sampled based on the similarity matrix and then reacted with the reactor to create a list of products, which are added to the input ``pandas.DataFrame``.


## reactor
This is the molecular transformer model used for reaction prediction.
This class is used to load the pre-trained molecular transformer model and create a Reactor instance as the reactor for ReactantPool.


-----------
written by Qi Zhang, 2021.01.16 updated by Stephen Wu, 2021.01.16
