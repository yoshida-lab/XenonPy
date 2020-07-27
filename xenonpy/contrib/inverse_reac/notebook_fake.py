#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 17:28:18 2020

@author: qi
"""
import os
os.chdir("/Users/qi/Documents/work/iSMD/inverse-reac")
import xenonpy
xenonpy.__version__
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#np.random.seed(201906)

# load in-house data
data = pd.read_csv("./STEREO_reactant_product_xlogp_tpsa_1000.csv")

# take a look at the data
data.columns
data.head()


# check target properties: E & HOMO-LUMO gap
plt.figure(figsize=(5,5))
plt.scatter(data['XLogP'],data['TPSA'],s=15,c='b',alpha = 0.1,label="full data")
plt.legend(loc='upper right')
plt.title('iSMD sample data')
plt.xlabel('XLogP')
plt.ylabel('topological polar surface area')
plt.show()

from xenonpy.descriptor import Fingerprints

RDKit_FPs = Fingerprints(featurizers=['ECFP', 'MACCS'], input_type='smiles')

tmp_FPs = RDKit_FPs.transform(data['product'])
print(tmp_FPs.head())


from ismd import GaussianLogLikelihood
from ismd import PoolSampler

# write down list of property name(s) for forward models and decide the target region
# (they will be used as a key in whole iQSPR run)
prop = ['XLogP','TPSA']
target_range = {'XLogP': (-2,2), 'TPSA': (0, 25)}

# import descriptor class to iQSPR and set the target of region of the properties
prd_mdls = GaussianLogLikelihood(descriptor=RDKit_FPs, targets = target_range)

# train forward models inside iQSPR
prd_mdls.fit(data['product'], data[prop])

pred = prd_mdls.predict(data['product'])
print(pred.head())

# calculate log-likelihood for a given target property region
tmp_ll = prd_mdls(data['product'], **target_range)
print(tmp_ll.head())

# plot histogram of log-likelihood values
tmp = tmp_ll.sum(axis = 1, skipna = True)

########################

cans = [smi for i, smi in enumerate(data['reactant'])
       if (data['XLogP'].iloc[i] > 4)]
init_samples = np.random.choice(cans, 10)
print(init_samples)


reactant_pool = [line.rstrip('\n') for line in open("/Users/qi/Documents/work/iSMD/data/STEREO_pool.txt")] #len(reactant_pool)=637645
reactant_pool1000=reactant_pool[:1000]
pool_sampler = PoolSampler(reactant_pool=reactant_pool1000)

n_loop = 5
for i in range(n_loop):
    init_samples = pool_sampler.proposal(init_samples)
    print('Round %i' % i,init_samples)


from ismd import Reactor
ChemicalReactor = Reactor()       
ChemicalReactor.BuildReactor(model_list=['./ismd/transformer_models/STEREO_mixed_augm_model_average_20.pt'], max_length=100, n_best=1) 

_, products = ChemicalReactor.react(init_samples)
tmp_ll = prd_mdls(products, **target_range)
tmp = tmp_ll.sum(axis = 1, skipna = True)

w = np.dot(tmp.values, 1)
w_sum = np.log(np.sum(np.exp(w - np.max(w)))) + np.max(w)  # avoid underflow
p = np.exp(w - w_sum)

np.random.choice(init_samples, size=len(init_samples), replace=True, p=p)

from ismd import ISMD

ismd_main = ISMD(estimator=prd_mdls, modifier=pool_sampler, reactor=ChemicalReactor)

for reac,product,tmp in ismd_main(init_samples,target_range):
    print(init_samples)
    print(product)
    print(tmp)











##############################








# N-gram library in XenonPy-iQSPR
from ismd import NGram

# initialize a new n-gram
n_gram = NGram()

# train the n-gram with SMILES of available molecules
n_gram.fit(data['product'],train_order=5)


np.random.seed(201903) # fix the random seed

# perform pure iQSPR molecule generation starting with 5 initial molecules
n_loop = 5
tmp = data['product'][:5]
for i in range(n_loop):
    tmp = n_gram.proposal(tmp)
    print('Round %i' % i,tmp)


# set up initial molecules for iQSPR
np.random.seed(201906) # fix the random seed
cans = [smi for i, smi in enumerate(data['reactant'])
       if (data['XLogP'].iloc[i] > 4)]
init_samples = np.random.choice(cans, 25)
print(init_samples)


# set up annealing schedule in iQSPR
beta = np.hstack([np.linspace(0.01,0.2,20),np.linspace(0.21,0.4,10),np.linspace(0.4,1,10),np.linspace(1,1,10)])
print('Number of steps: %i' % len(beta))
print(beta)


# library for running iQSPR in XenonPy-iQSPR
from ismd import ISMD

# update NGram parameters for this exampleHOMO-LUMO gap
n_gram.set_params(del_range=[1,20],max_len=500, reorder_prob=0.5, sample_order=(1,20))

# set up likelihood and n-gram models in iQSPR
iqspr_reorder = ISMD(estimator=prd_mdls, modifier=n_gram)
    
np.random.seed(201906) # fix the random seed
# main loop of iQSPR
iqspr_samples1, iqspr_loglike1, iqspr_prob1, iqspr_freq1 = [], [], [], []
for s, ll, p, freq in iqspr_reorder(init_samples, beta, yield_lpf=True):
    iqspr_samples1.append(s)
    iqspr_loglike1.append(ll)
    iqspr_prob1.append(p)
    iqspr_freq1.append(freq)
# record all outputs
iqspr_results_reorder = {
    "samples": iqspr_samples1,
    "loglike": iqspr_loglike1,
    "prob": iqspr_prob1,
    "freq": iqspr_freq1,
    "beta": beta
}

# save results
with open('iQSPR_results_reorder.obj', 'wb') as f:
    pk.dump(iqspr_results_reorder, f)
#############################


np.random.seed(201903) # fix the random seed

from ismd import PoolSampler

reactant_pool = [line.rstrip('\n') for line in open("/Users/qi/Documents/work/iSMD/data/STEREO_pool.txt")] #len(reactant_pool)=637645
reactant_pool1000=reactant_pool[:1000]
pool_sampler = PoolSampler(reactant_pool=reactant_pool1000)


# perform pure iQSPR molecule generation starting with 5 initial molecules
n_loop = 5
tmp = data_ss['reactant'][:5]
for i in range(n_loop):
    tmp = pool_sampler.proposal(tmp)
    print('Round %i' % i,tmp)
    





# library for running iQSPR in XenonPy-iQSPR
from xenonpy.contrib.ismd import BaseSMC_ismd

ismd_main = BaseSMC_ismd()
for reac in ismd_main(cans):
    print(reac)



















