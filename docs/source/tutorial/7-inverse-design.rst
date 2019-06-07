==============
Inverse design
==============

This tutorial provides step by step guidance to all the essential components for running iQSPR, an inverse molecular design algorithm based on machine learning. We provide a set of in-house data and pre-trained models for demonstration purpose. We reserve all rights for using these resources outside of this tutorial. We recommend readers to have prior knowledge about python programming, use of numpy and pandas packages, building models with scikit-learn, and understanding on the fundamental functions of XenonPy (refer to the tutorial on the descriptor calculation and model building with XenonPy). For any questions, please contact the developer of XenonPy.

--------
Overview
--------

We are interested in finding molecular structures :math:`S` such that their properties :math:`Y` have a high probability of falling into a target region :math:`U`, i.e., we want to sample from the posterior probability :math:`p(S|Y \in U)` that is proportional to :math:`p(Y \in U|S)p(S)` by the Bayes' theorem. :math:`p(Y \in U|S)` is the likelihood function that can be derived from any machine learning models predicting :math:`Y` for a given :math:`S`. :math:`p(S)` is the prior that represents all possible candidates of S to be considered. iQSPR is a numerical implementation for this Bayesian formulation based on sequential Monte Carlo sampling, which requires a likelihood model and a prior model, to begin with.

This tutorial will proceed as follow: (1) initial setup and data preparation, (2) descriptor preparation for the forward model (likelihood), (3) forward model (likelihood) preparation, (4) N-gram (prior) preparation, (5) a complete iQSPR run.

-------
Dataset
-------

We provide a data set that contains 16674 SMILES randomly selected from pubchem_ with 2 material properties, the internal energy E (kJ/mol) and the HOMO-LUMO gap (eV). You can download the file `here`_. The property values are obtained from single-point calculations in DFT (density functional theory) simulations, with compounds geometry optimized at the B3LYP/6-31G(d) level of theory using GAMESS. This data set is prepared by our previous developers of iQSPR. XenonPy supports pandas dataframe as the main input source. When there is miss match in the number of data points available in each material property, i.e., there exist missing values, please simply fill in the missing values with NaN, and XenonPy will automatically handle them during model training.

    >>> import pandas as pd
    >>> data = pd.read_csv("./iQSPR_sample_data.csv")

.. _here: https://github.com/yoshida-lab/XenonPy/releases/download/v0.3.1/iQSPR_sample_data.csv
.. _pubchem: https://pubchem.ncbi.nlm.nih.gov/

----------
Descriptor
----------

XenonPy provides out-of-the-box fingerprint calculators. We currently support all fingerprints and descriptors in the RDKit (Mordred will be added soon). In this tutorial, we only use the ECFP in RDKit. You may combine multiple descriptors as well. We currently support input_type to be 'smiles' or 'mols' (the RDKit internal mol format) and some of the basic options in RDKit fingerprints. The output will be a pandas dataframe that is supported scikit-learn when building a forward model with various machine learning methods.

    >>> from xenonpy.descriptor import Fingerprints
    >>> RDKit_FPs = Fingerprints(featurizers=['ECFP'], input_type='smiles')
    >>> calculated_FPs = RDKit_FPs.transform(data['SMILES'])

--------------------------
Forward model (likelihood)
--------------------------

The prepared descriptor class will be added to the forward model class used in iQSPR. To prepare the forward model, XenonPy provides two options: (1) use a template from XenonPy and train the model internally, and (2) prepare your own pre-trained model and feed it into the forward model class in iQSPR. Here, we will only show the first option.

The easiest way is to directly use the forward model template in XenonPy-iQSPR and train the model using the ``fit`` function.

    >>> from xenonpy.inverse.iqspr import GaussianLogLikelihood
    >>> prop = ['E','HOMO-LUMO gap']
    >>> prd_mdls = BayesianRidgeEstimator(descriptor=RDKit_FPs)
    >>> prd_mdls.fit(data['SMILES'], data[prop])

Once trained, we can evaluate the likelihood values for the molecules to hit a given target region in the property space. Note that the likelihood function is always in log-scale to avoid the numerical issue in the sampling step.

    >>> calculated_logL = prd_mdls.log_likelihood(data['SMILES'], **{'E': (0,200), 'HOMO-LUMO gap': (-np.inf, 3)})

    >>> from xenonpy.inverse.iqspr import GaussianLogLikelihood
    >>> prop = ['E','HOMO-LUMO gap']
    >>> prd_mdls = BayesianRidgeEstimator(descriptor=RDKit_FPs)
    >>> prd_mdls.fit(data['SMILES'], data[prop])

Once trained, we can evaluate the likelihood values for the molecules to hit a given target region in the property space. Note that the likelihood function is always in log-scale to avoid the numerical issue in the sampling step.

    >>> calculated_logL = prd_mdls.log_likelihood(data['SMILES'], **{'E': (0,200), 'HOMO-LUMO gap': (-np.inf, 3)})

    >>> from xenonpy.inverse.iqspr import BayesianRidgeEstimator
    >>> prop = ['E','HOMO-LUMO gap']
    >>> prd_mdls = BayesianRidgeEstimator(descriptor=RDKit_FPs)
    >>> prd_mdls.fit(data['SMILES'], data[prop])

Once trained, we can evaluate the likelihood values for the molecules to hit a given target region in the property space. Note that the likelihood function is always in log-scale to avoid the numerical issue in the sampling step.

    >>> calculated_logL = prd_mdls.log_likelihood(data['SMILES'], **{'E': (0,200), 'HOMO-LUMO gap': (-np.inf, 3)})

-------------------
NGram model (prior)
-------------------

For prior, which is simply a molecule generator, XenonPy currently provides N-gram model based on the extended SMILES language developed in Ikebata et al. [1]_

    >>> from xenonpy.inverse.iqspr import NGram
    >>> n_gram = NGram()
    >>> n_gram.fit(data['SMILES'],train_order=10)

Our N-gram-based molecular generator runs as follow: (1) given a tokenized SMILES in the extended SMILES format, randomly delete N tokens from the tail, (2) generate the next token using the N-gram table until we hit a termination token or a pre-set maximum length, (3) if generation ended without a termination token, a simple grammar check will be performed trying to fix any invalid parts, and the generated molecule will be abandoned if this step fails. Because we always start the modification from the tail and SMILES is not a 1-to-1 representation of a molecule, we recommend users to use the re-order function to randomly re-order the extended SMILES, so that you will not keep modifying the same part of the molecule. To do so, you can use the "set_params" function or do so when you initialize the N-gram using "NGram(...)". In fact, you can adjust other parameters in our N-gram model this way.

    >>> n_gram.set_params(del_range=[1,10],max_len=500, reorder_prob=0.5)

Having a good molecule generator is very important because it basically controls the search space of your inverse design. We recommend you to play with the different options available in the N-gram model before using it for the actual iQSPR run. For example, you can take a look at the molecules generated by the trained N-gram.

    >>> n_loop = 10
    >>> tmp = data['SMILES'][:10]
    >>> for i in range(n_loop):
    >>>     tmp = n_gram.proposal(tmp)
    >>>     print('Round %i' % i,tmp)

------
iQSPR
------

After the preparation of the forward model (likelihood) and N-gram model (prior), we are now ready to perform the actual iteration of iQSPR to generate molecules in our target property region.

We need to first set up some initial molecules as a starting point of our iQSPR iteration. Note that the number of molecules in this initial set governs the number of molecules generated in each iteration step. In practice, you may want at least 100 or even 1000 molecules per step depending your computing resources to avoid getting trapped in a local region when searching the whole molecular space defined by your N-gram model.

    >>> import numpy as np
    >>> init_samples = np.random.choice(data['SMILES'], 25)

For any sequential Monte Carlo algorithm, annealing is usually recommended to avoid getting trapped in a local mode. In iQSPR, we use the beta vector to control our annealing schedule. We recommend starting with a small number close to 0 to minimize the influence from the likelihood at the beginning steps and using some kind of exponential-like schedule to increase the beta value to 1, which represents the state of the original likelihood. The length of the beta vector directly controls the number of iteration in iQSPR. We recommend adding more steps with beta=1 at the end to allow exploration of the posterior distribution (your target property region). In practice, the iteration of the order of 100 or 1000 steps is recommended depending on your computing resources.

    >>> beta = np.hstack([np.linspace(0.01,0.2,20),np.linspace(0.21,0.4,10),np.linspace(0.4,1,10),np.linspace(1,1,10)])

Putting together the initial molecules, beta vector, forward model (likelihood), N-gram model (prior), you can now use a for-loop over the IQSPR class to get the generated molecules at each iteration step. More information can be extracted from the loop by setting "yield_lpf" to True (l: log-likelihood, p: probability of resampling, f: frequency of appearance). Note that the length of generated molecules in each step may not equal to the length of initial molecules because we only track the unique molecules and record their appearance frequency separately.

    >>> from xenonpy.inverse.iqspr import IQSPR
    >>> iqspr_reorder = IQSPR(estimator=prd_mdls, modifier=n_gram)
    >>> iqspr_samples1, iqspr_loglike1, iqspr_prob1, iqspr_freq1 = [], [], [], []
    >>> for s, ll, p, freq in iqspr_reorder(init_samples, beta, yield_lpf=True, **{'E': (0, 200), 'HOMO-LUMO gap': (-np.inf,3)}):
    >>>     iqspr_samples1.append(s)
    >>>     iqspr_loglike1.append(ll)
    >>>     iqspr_prob1.append(p)
    >>>     iqspr_freq1.append(freq)

Thank you for using XenonPy-iQSPR. We would appreciate any feedback and code contribution to this open-source project. For more details, you can check out our sample codes:

  https://github.com/yoshida-lab/XenonPy/tree/master/samples/iQSPR.ipynb

**Reference**

.. [1] Ikebata, H., Hongo, K., Isomura, T., Maezono, R. & Yoshida, R. Bayesian molecular design with a chemical language model. J. Comput. Aided. Mol. Des. 31, 379–391 (2017).