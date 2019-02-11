#  Copyright 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import re
from copy import deepcopy

# import necessary libraries
import numpy as np
import pandas as pd
import scipy.stats as sps
from numpy import random
from rdkit import Chem
from sklearn.base import RegressorMixin, TransformerMixin

from .base_smc import BaseSMC


class IQSPR(BaseSMC):
    def __init__(self, target, estimator, modifier):
        """
        SMC iQDPR runner.

        Parameters
        ----------
        target : object
        estimator : RegressorMixin
        modifier : TransformerMixin
        """
        if not isinstance(estimator, RegressorMixin):
            raise TypeError('<estimator> must be a subClass of  <BaseLogLikelihood>')
        if not isinstance(modifier, TransformerMixin):
            raise TypeError('<modifier> must be a subClass of  <BaseProposer>')
        self._modifier = modifier
        self._estimator = estimator
        self.target = target

    @property
    def modifier(self):
        return self._modifier

    @modifier.setter
    def modifier(self, value):
        if not isinstance(value, RegressorMixin):
            raise TypeError('<modifier> must implement  <RegressorMixin>')
        self._modifier = value

    @property
    def estimator(self):
        return self._estimator

    @estimator.setter
    def estimator(self, value):
        if not isinstance(value, TransformerMixin):
            raise TypeError('<modifier> must immplement  <TransformerMixin>')
        self._estimator = value

    @classmethod
    def smi2list(cls, smi_str):
        # smi_pat = '(=\[.*?\]|#\[.*?\]|\[.*?\]|=Br|#Br|=Cl|#Cl|Br|Cl|=.|#.|\%[0-9][0-9]|\w|\W)'
        smi_pat = r'(\[.*?\]|Br|Cl|\%[0-9][0-9]|\w|\W)'
        smi_list = list(filter(None, re.split(smi_pat, smi_str)))
        return smi_list

    @classmethod
    def smi2esmi(cls, smi_str):
        smi_list = cls.smi2list(smi_str)

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
                v_substr.append(deepcopy(tmp_ss))  # contracted substring added w/o ')'
                c_br = 0
            elif c_br == 1:
                c_br = 2

            if esmi_list[i] == '(':
                c_br = 1
                n_br += 1
            elif esmi_list[i] == ')':
                tmp_ss = deepcopy(v_substr[-1])  # retrieve contracted substring added w/o ')'
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
            substr_list.append(deepcopy(tmp_ss))
            br_list.append(n_br)
            ring_list.append(len(v_ringn))

        return pd.DataFrame({'esmi': esmi_list, 'n_br': br_list, 'n_ring': ring_list, 'substr': substr_list})

    # may add error check here in the future?
    @classmethod
    def esmi2smi(cls, esmi_pd):
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

    def translator(self, x, inverse=False):
        if inverse:
            return [self.esmi2smi(x_) for x_ in x]
        return [self.smi2esmi(x_) for x_ in x]

    def log_likelihood(self, x):
        # target molecule with Tg > tar_min_Tg

        ll = np.repeat(-1000.0, len(x))  # note: the constant will determine the vector type!
        mols = []
        idx = []
        for i in range(len(x)):
            try:
                mols.append(Chem.MolFromSmiles(self.esmi2smi(x[i])))
                idx.append(i)
            except BaseException:
                pass
        # convert extended SMILES to fingerprints
        tar_fps = self.descriptor_gen.transform(mols)
        tmp = tar_fps.isna().any(axis=1)
        idx = [idx[i] for i in range(len(idx)) if ~tmp[i]]
        tar_fps.dropna(inplace=True)
        # predict Tg values and calc. log-likelihood
        tar_mean, tar_std = self._estimator.predict(tar_fps, return_std=True)
        tmp = sps.norm.logcdf(-self.target, loc=-np.asarray(tar_mean), scale=np.asarray(tar_std))
        np.put(ll, idx, tmp)
        return ll

    def update_ngram(self, esmi_pd):
        for iB in [False, True]:
            # index for open/closed branches char. position, remove last row for '!'
            idx_B = esmi_pd.iloc[:-1].index[(esmi_pd['n_br'].iloc[:-1] > 0) == iB]
            list_R = esmi_pd['n_ring'][idx_B].unique().tolist()
            if len(list_R) > 0:
                if len(self._modifier[0][iB]) < (max(list_R) + 1):  # expand list of dataframe for max. num-of-ring + 1
                    for ii in range(len(self._modifier)):
                        self._modifier[ii][iB].extend(
                            [pd.DataFrame() for i in range((max(list_R) + 1) - len(self._modifier[ii][iB]))])
                for iR in list_R:
                    idx_R = idx_B[esmi_pd['n_ring'][idx_B] == iR]  # index for num-of-open-ring char. pos.
                    tar_char = esmi_pd['esmi'][
                        idx_R + 1].tolist()  # shift one down for 'next character given substring'
                    tar_substr = esmi_pd['substr'][idx_R].tolist()
                    for iO in range(len(self._modifier)):
                        idx_O = [x for x in range(len(tar_substr)) if
                                 len(tar_substr[x]) > iO]  # index for char with substring length not less than order
                        for iC in idx_O:
                            if not tar_char[iC] in self._modifier[iO][iB][iR].columns.tolist():
                                self._modifier[iO][iB][iR][tar_char[iC]] = 0
                            tmp_row = str(tar_substr[iC][-(iO + 1):])
                            if not tmp_row in self._modifier[iO][iB][iR].index.tolist():
                                self._modifier[iO][iB][iR].loc[tmp_row] = 0
                            self._modifier[iO][iB][iR].loc[
                                tmp_row, tar_char[iC]] += 1  # somehow 'at' not ok with mixed char and int column names

        # return self._ngram_tab #maybe not needed?

    def proposal(self, x, size, p=None):
        if self.n_gram_table is None:
            raise ValueError(
                'Must have a pre-trained n-gram table,',
                'you can set one your already had or train a new one by using <update_ngram> method')
        pass

    # get probability vector for sampling next character, return character list and corresponding probability in numpy.array (normalized)
    # may cause error if empty string list is fed into 'tmp_str'
    # Warning: maybe can reduce the input of iB and iR - directly input the reduced list of self._ngram_tab (?)
    # Warning: may need to update this function with bisection search for faster speed (?)
    # Warning: may need to add worst case that no pattern found at all?
    def get_prob(self, tmp_str, iB, iR):
        # right now we use back-off method, an alternative is Kneser–Nay smoothing
        for iO in range(len(self._modifier) - 1, -1, -1):
            if (len(tmp_str) > iO) & (str(tmp_str[-(iO + 1):]) in self._modifier[iO][iB][iR].index.tolist()):
                cand_char = self._modifier[iO][iB][iR].columns.tolist()
                cand_prob = np.array(self._modifier[iO][iB][iR].loc[str(tmp_str[-(iO + 1):])])
                break
        return (cand_char, cand_prob / sum(cand_prob))

    # get the next character, return the probability value
    def sample_next_char(self, esmi_pd):
        iB = esmi_pd['n_br'].iloc[-1] > 0
        iR = esmi_pd['n_ring'].iloc[-1]
        cand_char, cand_prob = self.get_prob(esmi_pd['substr'].iloc[-1], self._modifier, iB, iR)
        # here we assume cand_char is not empty
        tmp = random.choices(range(len(cand_char)), weights=cand_prob)
        esmi_pd = self.add_char(esmi_pd, cand_char[tmp[0]])
        return (esmi_pd, cand_prob[tmp[0]])

    def add_char(esmi_pd, next_char):
        new_pd_row = esmi_pd.iloc[-1]
        new_pd_row.at['substr'] = new_pd_row['substr'] + [next_char]
        new_pd_row.at['esmi'] = next_char
        if next_char == '(':
            new_pd_row.at['n_br'] += 1
        elif next_char == ')':
            new_pd_row.at['n_br'] -= 1
            #        # assume '(' must exist before if the extended SMILES is valid! (will fail if violated)
            #        idx = next((x for x in range(len(new_pd_row['substr'])-1,-1,-1) if new_pd_row['substr'][x] == '('), None)
            # find index of the last unclosed '('
            tmp_c = 1
            for x in range(len(new_pd_row['substr']) - 2, -1, -1):  # exclude the already added "next_char"
                if new_pd_row['substr'][x] == '(':
                    tmp_c -= 1
                elif new_pd_row['substr'][x] == ')':
                    tmp_c += 1
                if tmp_c == 0:
                    idx = x
                    break
            # assume no '()' and '((' pattern that is not valid/possible in SMILES
            new_pd_row.at['substr'] = new_pd_row['substr'][:(idx + 2)] + [')']
        elif next_char == '&':
            new_pd_row.at['n_ring'] += 1
        elif isinstance(next_char, int):
            new_pd_row.at['n_ring'] -= 1
        return esmi_pd.append(new_pd_row, ignore_index=True)

    def del_char(esmi_pd, n_char):
        return esmi_pd[:-n_char]

    # need to make sure esmi_pd is a completed SMILES to use this function
    def reorder_esmi(self, esmi_pd):
        # convert back to SMILES first, then to rdkit MOL
        m = Chem.MolFromSmiles(self.esmi2smi(esmi_pd))
        idx = random.choice(range(m.GetNumAtoms()))
        # currently assume kekuleSmiles=True, i.e., no small letters but with ':' for aromatic rings
        esmi_pd = self.smi2esmi(Chem.MolToSmiles(m, rootedAtAtom=idx, kekuleSmiles=True))
        return esmi_pd

    def valid_esmi(self, esmi_pd):
        # delete all ending '(' or '&'
        for i in range(len(esmi_pd)):
            if not ((esmi_pd['esmi'].iloc[-1] == '(') | (esmi_pd['esmi'].iloc[-1] == '&')):
                break
            esmi_pd = self.del_char(esmi_pd, 1)
        # delete or fill in ring closing
        flag_ring = esmi_pd['n_ring'].iloc[-1] > 0
        for i in range(len(esmi_pd)):  # max to double the length of current SMILES
            if flag_ring and (random.random() < 0.7):  # 50/50 for adding two new char.
                # add a character
                esmi_pd, _ = self.sample_next_char(esmi_pd, self._modifier)
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
            esmi_pd = self.smi2esmi(self.esmi2smi(esmi_pd.drop(esmi_pd.index[num_open]).reset_index(drop=True)))
        #    if esmi_pd['n_ring'].iloc[-1] > 0:
        #        if random.getrandbits(1): # currently 50% change adding
        #            # add a character
        #            esmi_pd, _ = sample_next_char(esmi_pd,self._ngram_tab)
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
            esmi_pd = self.add_char(esmi_pd, ')')
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

    def mod_esmi(self, esmi_pd, n=8, p=0.5):
        # esmi_pd = reorder_esmi(esmi_pd)
        # number of add/delete (n) with probability of add = p
        n_add = sum(np.random.choice([False, True], n, p=[1 - p, p]))
        # first delete then add
        esmi_pd = self.del_char(esmi_pd, min(n - n_add + 1, len(esmi_pd) - 1))  # at least leave 1 character
        for i in range(n_add):
            esmi_pd, _ = self.sample_next_char(esmi_pd, self._modifier)
            if esmi_pd['esmi'].iloc[-1] == '!':
                return esmi_pd  # stop when hitting '!', assume must be valid SMILES
        print(self.esmi2smi(esmi_pd))
        print(esmi_pd)
        print("-----")
        esmi_pd = self.valid_esmi(esmi_pd, self._modifier)
        new_pd_row = {'esmi': '!', 'n_br': 0, 'n_ring': 0, 'substr': esmi_pd['substr'].iloc[-1] + ['!']}
        return esmi_pd.append(new_pd_row, ignore_index=True)
