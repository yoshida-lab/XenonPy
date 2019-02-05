# Copyright 2019 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import scipy.stats as sps
from rdkit import Chem
from sklearn.linear_model import BayesianRidge

from xenonpy.descriptor import BaseDescriptor
from xenonpy.descriptor import MorganFingerprint as FPsCalc


class RdkitDesc_MOLS(BaseDescriptor):
    def __init__(self, n_jobs=-1, *, elements=None, include=None,
                 exclude=None):
        super().__init__()
        self.n_jobs = n_jobs

        self.rdkit_fp = FPsCalc(n_jobs)
        # self.rdkit_fp = MACCS_MOLS(n_jobs)


mols = [Chem.MolFromSmiles(x) for x in data_PG['SMILES']]
ss = pd.Series(mols, name='rdkit_fp', index=data_PG['PGID_'])
rdkit_fp = RdkitDesc_MOLS()
PG_fps = rdkit_fp.fit_transform(ss)

x_train = PG_fps
y_train = data_PG['Glass-Transition-Temperature']
mdl = BayesianRidge(compute_score=True)
mdl.fit(x_train, y_train)
prd_train, std_train = mdl.predict(x_train, return_std=True)


# prepare XenonPy fingerprint calculator

# setup likelihood function for iqspr
def logLikelihood(esmi_pds):
    # target molecule with Tg > tar_min_Tg
    tar_min_Tg = 400
    # convert extended SMILES to fingerprints
    mols = [Chem.MolFromSmiles(esmi2smi(x)) for x in esmi_pds]
    ss = pd.Series(mols, name='rdkit_fp')
    rdkit_fp = RdkitDesc_MOLS()
    tar_fps = rdkit_fp.fit_transform(ss)
    # predict Tg values and calc. log-likelihood
    tar_mean, tar_std = mdl.predict(tar_fps, return_std=True)
    ll = sps.norm.logcdf(-tar_min_Tg, loc=-np.asarray(tar_mean), scale=np.asarray(tar_std))
    return ll


import copy
import random
import re


def smi2list(smi_str):
    # smi_pat = '(=\[.*?\]|#\[.*?\]|\[.*?\]|=Br|#Br|=Cl|#Cl|Br|Cl|=.|#.|\%[0-9][0-9]|\w|\W)'
    smi_pat = '(\[.*?\]|Br|Cl|\%[0-9][0-9]|\w|\W)'
    smi_list = list(filter(None, re.split(smi_pat, smi_str)))
    return smi_list


def smi2esmi(smi_str):
    smi_list = smi2list(smi_str)

    esmi_list = smi_list + ['!']
    substr_list = []  # list of all contracted substrings (include current char.)
    br_list = []  # list of whether open branch exist at current character position (include current char.)
    ring_list = []  # list of number of open ring at current character position (include current char.)
    v_substr = []  # list of temporary contracted substrings
    v_ringn = []  # list of numbering of open rings
    c_br = 0  # tracking open branch steps for recording contracted substrings
    n_br = 0  # tracking number of open branches
    tmp_ss = []  # list of current contracted substring
    for i in range(len(esmi_list)):
        if c_br == 2:
            v_substr.append(copy.deepcopy(tmp_ss))  # contracted substring added w/o ')'
            c_br = 0
        elif c_br == 1:
            c_br = 2

        if esmi_list[i] == '(':
            c_br = 1
            n_br += 1
        elif esmi_list[i] == ')':
            tmp_ss = copy.deepcopy(v_substr[-1])  # retrieve contracted substring added w/o ')'
            v_substr.pop()
            n_br -= 1
        elif '%' in esmi_list[i]:
            esmi_list[i] = int(esmi_list[i][1:3])
            if esmi_list[i] in v_ringn:
                esmi_list[i] = v_ringn.index(esmi_list[i])
                v_ringn.pop(esmi_list[i])
            else:
                v_ringn.append(esmi_list[i])
                esmi_list[i] = '&'
        elif esmi_list[i].isdigit():
            esmi_list[i] = int(esmi_list[i])
            if esmi_list[i] in v_ringn:
                esmi_list[i] = v_ringn.index(esmi_list[i])
                v_ringn.pop(esmi_list[i])
            else:
                v_ringn.append(esmi_list[i])
                esmi_list[i] = '&'

        tmp_ss.append(esmi_list[i])
        substr_list.append(copy.deepcopy(tmp_ss))
        br_list.append(n_br)
        ring_list.append(len(v_ringn))

    esmi_pd = pd.DataFrame({'esmi': esmi_list, 'n_br': br_list, 'n_ring': ring_list, 'substr': substr_list})
    return esmi_pd


# may add error check here in the future?
def esmi2smi(esmi_pd):
    smi_list = esmi_pd['esmi'].tolist()
    num_open = []
    num_unused = list(range(99, 0, -1))
    for i in range(len(smi_list)):
        if smi_list[i] == '&':
            if num_unused[-1] > 9:
                smi_list[i] = ''.join(['%', str(num_unused[-1])])
            else:
                smi_list[i] = str(num_unused[-1])
            num_open.insert(0, num_unused[-1])
            num_unused.pop()
        elif isinstance(smi_list[i], int):
            tmp = int(smi_list[i])
            if num_open[tmp] > 9:
                smi_list[i] = ''.join(['%', str(num_open[tmp])])
            else:
                smi_list[i] = str(num_open[tmp])
            num_unused.append(num_open[tmp])
            num_open.pop(tmp)
    if smi_list[-1] == "!":  # cover cases of incomplete esmi_pd
        smi_list.pop()  # remove the final '!'
    smi_str = ''.join(smi_list)
    return smi_str


# expect ngram_tab to be a list of pandas tables up to max order, and esmi_pd in valid form (last row is '!')
def update_ngram(ngram_tab, esmi_pd):
    for iB in [False, True]:
        # index for open/closed branches char. position, remove last row for '!'
        idx_B = esmi_pd.iloc[:-1].index[(esmi_pd['n_br'].iloc[:-1] > 0) == iB]
        list_R = esmi_pd['n_ring'][idx_B].unique().tolist()
        if len(list_R) > 0:
            if len(ngram_tab[0][iB]) < (max(list_R) + 1):  # expand list of dataframe for max. num-of-ring + 1
                for ii in range(len(ngram_tab)):
                    ngram_tab[ii][iB].extend(
                        [pd.DataFrame() for i in range((max(list_R) + 1) - len(ngram_tab[ii][iB]))])
            for iR in list_R:
                idx_R = idx_B[esmi_pd['n_ring'][idx_B] == iR]  # index for num-of-open-ring char. pos.
                tar_char = esmi_pd['esmi'][idx_R + 1].tolist()  # shift one down for 'next character given substring'
                tar_substr = esmi_pd['substr'][idx_R].tolist()
                for iO in range(len(ngram_tab)):
                    idx_O = [x for x in range(len(tar_substr)) if
                             len(tar_substr[x]) > iO]  # index for char with substring length not less than order
                    for iC in idx_O:
                        if not tar_char[iC] in ngram_tab[iO][iB][iR].columns.tolist():
                            ngram_tab[iO][iB][iR][tar_char[iC]] = 0
                        tmp_row = str(tar_substr[iC][-(iO + 1):])
                        if not tmp_row in ngram_tab[iO][iB][iR].index.tolist():
                            ngram_tab[iO][iB][iR].loc[tmp_row] = 0
                        ngram_tab[iO][iB][iR].loc[
                            tmp_row, tar_char[iC]] += 1  # somehow 'at' not ok with mixed char and int column names
    # return ngram_tab #maybe not needed?


# get probability vector for sampling next character, return character list and corresponding probability in numpy.array (normalized)
# may cause error if empty string list is fed into 'tmp_str'
# Warning: maybe can reduce the input of iB and iR - directly input the reduced list of ngram_tab (?)
# Warning: may need to update this function with bisection search for faster speed (?)
# Warning: may need to add worst case that no pattern found at all?
def get_prob(tmp_str, ngram_tab, iB, iR):
    # right now we use back-off method, an alternative is Kneser–Nay smoothing
    for iO in range(len(ngram_tab) - 1, -1, -1):
        if (len(tmp_str) > iO) & (str(tmp_str[-(iO + 1):]) in ngram_tab[iO][iB][iR].index.tolist()):
            cand_char = ngram_tab[iO][iB][iR].columns.tolist()
            cand_prob = np.array(ngram_tab[iO][iB][iR].loc[str(tmp_str[-(iO + 1):])])
            break
    return (cand_char, cand_prob / sum(cand_prob))


# get the next character, return the probability value
def sample_next_char(esmi_pd, ngram_tab):
    iB = esmi_pd['n_br'].iloc[-1] > 0
    iR = esmi_pd['n_ring'].iloc[-1]
    cand_char, cand_prob = get_prob(esmi_pd['substr'].iloc[-1], ngram_tab, iB, iR)
    # here we assume cand_char is not empty
    tmp = random.choices(range(len(cand_char)), weights=cand_prob)
    esmi_pd = add_char(esmi_pd, cand_char[tmp[0]])
    return (esmi_pd, cand_prob[tmp[0]])


def add_char(esmi_pd, next_char):
    new_pd_row = esmi_pd.iloc[-1]
    new_pd_row.at['substr'] = new_pd_row['substr'] + [next_char]
    new_pd_row.at['esmi'] = next_char
    if next_char == '(':
        new_pd_row.at['n_br'] += 1
    elif next_char == ')':
        new_pd_row.at['n_br'] -= 1
        # assume '(' must exist before if the extended SMILES is valid! (will fail if violated)
        idx = next((x for x in range(len(new_pd_row['substr']) - 1, -1, -1) if new_pd_row['substr'][x] == '('), None)
        # assume no '()' pattern that is not valid in SMILES
        new_pd_row.at['substr'] = new_pd_row['substr'][:(idx + 2)] + [')']
    elif next_char == '&':
        new_pd_row.at['n_ring'] += 1
    elif isinstance(next_char, int):
        new_pd_row.at['n_ring'] -= 1
    return esmi_pd.append(new_pd_row, ignore_index=True)


def del_char(esmi_pd, n_char):
    return esmi_pd[:-n_char]


# need to make sure esmi_pd is a completed SMILES to use this function
def reorder_esmi(esmi_pd):
    # convert back to SMILES first, then to rdkit MOL
    m = Chem.MolFromSmiles(esmi2smi(esmi_pd))
    idx = random.choice(range(len(m.GetAtoms())))
    # currently assume kekuleSmiles=True, i.e., no small letters but with ':' for aromatic rings
    esmi_pd = smi2esmi(Chem.MolToSmiles(m, rootedAtAtom=idx, kekuleSmiles=True))
    return esmi_pd


def valid_esmi(esmi_pd, ngram_tab):
    # delete all ending '(' or '&'
    for i in range(len(esmi_pd)):
        if not ((esmi_pd['esmi'].iloc[-1] == '(') | (esmi_pd['esmi'].iloc[-1] == '&')):
            break
        esmi_pd = del_char(esmi_pd, 1)
    # delete or fill in ring closing
    flag_ring = esmi_pd['n_ring'].iloc[-1] > 0
    for i in range(len(esmi_pd)):  # max to double the length of current SMILES
        if flag_ring and (random.random() < 0.7):  # 50/50 for adding two new char.
            # add a character
            esmi_pd, _ = sample_next_char(esmi_pd, ngram_tab)
            flag_ring = esmi_pd['n_ring'].iloc[-1] > 0
        else:
            break
    if flag_ring:
        # prepare for delete (1st letter shall not be '&')
        tmp_idx = esmi_pd.iloc[1:].index
        tmp_count = np.array(esmi_pd['n_ring'].iloc[1:]) - np.array(esmi_pd['n_ring'].iloc[:-1])
        num_open = tmp_idx[tmp_count == 1]
        num_close = esmi_pd['esmi'][tmp_count == -1]
        for i in num_close:
            num_open.pop(i)
        # delete all irrelevant rows and reconstruct esmi
        esmi_pd = smi2esmi(esmi2smi(esmi_pd.drop(esmi_pd.index[num_open]).reset_index(drop=True)))
    #    if esmi_pd['n_ring'].iloc[-1] > 0:
    #        if random.getrandbits(1): # currently 50% change adding
    #            # add a character
    #            esmi_pd, _ = sample_next_char(esmi_pd,ngram_tab)
    #        else:
    #            # prepare for delete (1st letter shall not be '&')
    #            tmp_idx = esmi_pd.iloc[1:].index
    # #            tmp_ring = np.array(esmi_pd['n_ring'].iloc[1:]) - np.array(esmi_pd['n_ring'].iloc[:-1]) != 0
    # #            num_open = [] #record index of unclosed ring symbols
    # #            for i in tmp_idx[tmp_ring]:
    # #               if esmi_pd.loc[i,'esmi'] == '&':
    # #                   num_open.insert(0,i)
    # #               else:
    # #                   num_open.pop(esmi_pd.loc[i,'esmi'])
    #            tmp_count = np.array(esmi_pd['n_ring'].iloc[1:]) - np.array(esmi_pd['n_ring'].iloc[:-1])
    #            num_open = tmp_idx[tmp_count == 1]
    #            num_close = esmi_pd['esmi'][tmp_count == -1]
    #            for i in num_close:
    #                num_open.pop(i)
    #            # delete all irrelevant rows and reconstruct esmi
    #            esmi_pd = smi2esmi(esmi2smi(esmi_pd.drop(esmi_pd.index[num_open]).reset_index(drop=True)))
    # esmi_pd = esmi_pd.iloc[:-1]
    # fill in branch closing (last letter shall not be '(')
    for i in range(esmi_pd['n_br'].iloc[-1]):
        esmi_pd = add_char(esmi_pd, ')')
    #    # change back some lower letter to upper letter (this is wrong cause letter before ring starts will also get converted)
    #    tmp_ring = (np.array(esmi_pd['n_ring'].iloc[:-1]) + np.array(esmi_pd['n_ring'].iloc[1:])) == 0
    #    tmp_low = np.array([x.islower() if not isinstance(x, int) else False for x in esmi_pd['esmi'].iloc[:-1]])
    #    tmp_idx = esmi_pd.iloc[:-1].index
    #    esmi_pd.loc[tmp_idx[tmp_low & tmp_ring],'esmi'] = esmi_pd['esmi'][tmp_idx[tmp_low & tmp_ring]].str.upper()
    #    if not isinstance(esmi_pd['esmi'].iloc[-1],int):
    #        if (esmi_pd['esmi'].iloc[-1].islower()) & (esmi_pd['n_ring'].iloc[-1] == 0):
    #            #will have warning here, but no solution found yet... both .loc or .at doesn't work (add '-1' row)
    #            esmi_pd['esmi'].iloc[-1] = esmi_pd['esmi'].iloc[-1].upper()
    return esmi_pd


def mod_esmi(esmi_pd, ngram_tab, n=8, p=0.5):
    # esmi_pd = reorder_esmi(esmi_pd)
    # number of add/delete (n) with probability of add = p
    n_add = sum(np.random.choice([False, True], n, p=[1 - p, p]))
    # first delete then add
    esmi_pd = del_char(esmi_pd, min(n - n_add + 1, len(esmi_pd) - 1))  # at least leave 1 character
    for i in range(n_add):
        esmi_pd, _ = sample_next_char(esmi_pd, ngram_tab)
        if esmi_pd['esmi'].iloc[-1] == '!':
            return esmi_pd  # stop when hitting '!', assume must be valid SMILES
    print(esmi2smi(esmi_pd))
    print(esmi_pd)
    print("-----")
    esmi_pd = valid_esmi(esmi_pd, ngram_tab)
    new_pd_row = {'esmi': '!', 'n_br': 0, 'n_ring': 0, 'substr': esmi_pd['substr'].iloc[-1] + ['!']}
    return esmi_pd.append(new_pd_row, ignore_index=True)


order = 10
ngram_tab = [[[], []] for i in range(order)]  # initialize 'order' copies of list with 2 empty lists in it

cans = [Chem.MolToSmiles(Chem.MolFromSmiles(smi), kekuleSmiles=True) for smi in data_PG['SMILES']]

for smi in cans:
    esmi_pd = smi2esmi(smi)
    update_ngram(ngram_tab, esmi_pd)

with open('PG_ngram_O10.obj', 'wb') as f:
    pk.dump(ngram_tab, f)

for iO in range(len(ngram_tab)):
    for iB in range(len(ngram_tab[iO])):
        for iR in range(len(ngram_tab[iO][iB])):
            print(iO, iB, iR)
            print(ngram_tab[iO][iB][iR])
            print('-----')

np.random.seed(3458)

n = 10
beta = np.linspace(0.05, 1, 10)
ini_smis = [cans[x] for x in range(len(cans))
            if cans[x][0] == '*'
            and data_PG['Glass-Transition-Temperature'].iloc[x] < 300]
s0 = np.random.choice(ini_smis, n)

# pandas values_count maybe faster for n > 5000
_, unq_idx, unq_cnt = np.unique(s0, return_index=True, return_counts=True)
s = [smi2esmi(s0[x]) for x in unq_idx]
for i in range(len(beta)):
    print(i)
    print([esmi2smi(x) for x in s])

    w = logLikelihood(s) * beta[i] + np.log(unq_cnt)  # annealed likelihood in log - adjust with copy counts
    wSum = np.log(sum(np.exp(w - max(w)))) + max(w)  # avoid underflow

    idx = np.random.choice(len(w), n, p=np.exp(w - wSum))

    s = [mod_esmi(s[x], ngram_tab) for x in idx]
    # take only unique copies and update unq_cnt
    _, unq_idx, unq_cnt = np.unique([str(x['esmi'].tolist()) for x in s], return_index=True, return_counts=True)
    s = [s[x] for x in unq_idx]

# return s
print(np.exp(likelihood_fcn(s)))
print([esmi2smi(x) for x in s])
