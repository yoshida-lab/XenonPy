#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import re
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from xenonpy.inverse.base import BaseProposal, ProposalError


class GetProbError(ProposalError):

    def __init__(self, tmp_str, i_b, i_r):
        self.tmp_str = tmp_str
        self.iB = i_b
        self.iR = i_r
        self.old_smi = None

        super().__init__('get_prob: %s not found in NGram, iB=%i, iR=%i' % (tmp_str, i_b, i_r))


class MolConvertError(ProposalError):
    def __init__(self, new_smi):
        self.new_smi = new_smi
        self.old_smi = None

        super().__init__('can not convert %s to Mol' % new_smi)


class NGramTrainingError(ProposalError):
    def __init__(self, error, smi):
        self.old_smi = smi

        super().__init__('training failed for %s, because of <%s>: %s' % (smi, error.__class__.__name__, error))


class NGram(BaseProposal):
    def __init__(self, *, ngram_tab=None, sample_order=(1, 10), del_range=(1, 10), min_len = 1, max_len=1000, reorder_prob=0):
        """
        N-Garm

        Parameters
        ----------
        ngram_tab: NGram table
            NGram table for modify SMILES.
        sample_order: tuple[int, int] or int
            range of order of ngram table used during proposal,
            when given int, sample_order = (1, int)
        del_range: tuple[int, int] or int
            range of random deletion of SMILES string during proposal,
            when given int, del_range = (1, int)
        min_len: int
            minimum length of the extended SMILES,
            shall be smaller than the lower bound of the sample_order
        max_len: int
            max length of the extended SMILES to be terminated from continuing modification
        reorder_prob: float
            probability of the SMILES being reordered during proposal
        """

        self.sample_order = sample_order
        self.reorder_prob = reorder_prob
        self.min_len = min_len
        self.max_len = max_len
        self.del_range = del_range

        if ngram_tab is None:
            self._table = None
            self._train_order = None
        else:
            self._table = deepcopy(ngram_tab)
            self._train_order = (1, len(ngram_tab))

        self._fit_sample_order()
        self._fit_min_len()

    @property
    def sample_order(self):
        return self._sample_order

    @sample_order.setter
    def sample_order(self, val):
        if isinstance(val, int):
            self._sample_order = (1, val)
        elif isinstance(val, tuple):
            self._sample_order = val
        elif isinstance(val, (list, np.array, pd.Series)):
            self._sample_order = (val[0], val[1])
        else:
            raise TypeError('please input a <tuple> of two <int> or a single <int> for sample_order')
        if self._sample_order[0] < 1:
            raise RuntimeError('Min sample_order must be greater than 0')
        if self._sample_order[1] < self._sample_order[0]:
            raise RuntimeError('Min sample_order must not be smaller than max sample_order')

    @property
    def reorder_prob(self):
        return self._reorder_prob

    @reorder_prob.setter
    def reorder_prob(self, val):
        if isinstance(val, (int, float)):
            self._reorder_prob = val
        else:
            raise TypeError('please input a <float> for reorder_prob')

    @property
    def min_len(self):
        return self._min_len

    @min_len.setter
    def min_len(self, val):
        if isinstance(val, int):
            self._min_len = val
        else:
            raise TypeError('please input a <int> for min_len')

    @property
    def max_len(self):
        return self._max_len

    @max_len.setter
    def max_len(self, val):
        if isinstance(val, int):
            self._max_len = val
        else:
            raise TypeError('please input a <int> for max_len')

    @property
    def del_range(self):
        return self._del_range

    @del_range.setter
    def del_range(self, val):
        if isinstance(val, int):
            self._del_range = (1, val)
        elif isinstance(val, tuple):
            self._del_range = val
        elif isinstance(val, (list, np.array, pd.Series)):
            self._del_range = (val[0], val[1])
        else:
            raise TypeError('please input a <tuple> of two <int> or a single <int> for del_range')
        if self._del_range[1] < self._del_range[0]:
            raise RuntimeError('Min del_range must not be smaller than max del_range')

    def _fit_sample_order(self):
        if self._train_order and self._train_order[1] < self.sample_order[1]:
            warnings.warn('max <sample_order>: %s is greater than max <train_order>: %s,'
                          'max <sample_order> will be reduced to max <train_order>' % (self.sample_order[1], self._train_order[1]),
                          RuntimeWarning)
            self.sample_order = (self.sample_order[0], self._train_order[1])
        if self._train_order and self._train_order[0] > self.sample_order[0]:
            warnings.warn('min <sample_order>: %s is smaller than min <train_order>: %s,'
                          'min <sample_order> will be increased to min <train_order>' % (self.sample_order[0], self._train_order[0]),
                          RuntimeWarning)
            self.sample_order = (self._train_order[0], self.sample_order[1])

    def _fit_min_len(self):
        if self.sample_order[0] > self.min_len:
            warnings.warn('min <sample_order>: %s is greater than min_len: %s,'
                          'min_len will be increased to min <sample_order>' % (
                          self.sample_order[0], self.min_len),
                          RuntimeWarning)
            self.min_len = self.sample_order[0]

    def on_errors(self, error):
        """

        Parameters
        ----------
        error: ProposalError
            Error object.
        Returns
        -------

        """
        if isinstance(error, GetProbError):
            return error.old_smi
        if isinstance(error, MolConvertError):
            return error.old_smi
        if isinstance(error, NGramTrainingError):
            pass

    @property
    def ngram_table(self):
        return deepcopy(self._table)

    @ngram_table.setter
    def ngram_table(self, value):
        self._table = deepcopy(value)

    def modify(self, ext_smi):
        # reorder for a given probability
        if np.random.random() < self.reorder_prob:
            ext_smi = self.reorder_esmi(ext_smi)
        # number of deletion (randomly pick from given range)
        n_del = np.random.randint(self.del_range[0], self.del_range[1] + 1)
        # first delete then add
        ext_smi = self.del_char(ext_smi, min(n_del + 1, len(ext_smi) - self.min_len))  # at least leave min_len char
        # add until reaching '!' or a given max value
        for i in range(self.max_len - len(ext_smi)):
            ext_smi, _ = self.sample_next_char(ext_smi)
            if ext_smi['esmi'].iloc[-1] == '!':
                return ext_smi  # stop when hitting '!', assume must be valid SMILES
        # check incomplete esmi
        ext_smi = self.validator(ext_smi)
        # fill in the '!'
        new_pd_row = {'esmi': '!', 'n_br': 0, 'n_ring': 0, 'substr': ext_smi['substr'].iloc[-1] + ['!']}

        warnings.warn('Extended SMILES hits max length', RuntimeWarning)

        return ext_smi.append(new_pd_row, ignore_index=True)

    @classmethod
    def smi2list(cls, smiles):
        # smi_pat = r'(-\[.*?\]|=\[.*?\]|#\[.*?\]|\[.*?\]|-Br|=Br|#Br|-Cl|=Cl|#Cl|Br|Cl|-.|=.|#.|\%[0-9][0-9]|\w|\W)'
        # smi_pat = r'(=\[.*?\]|#\[.*?\]|\[.*?\]|=Br|#Br|=Cl|#Cl|Br|Cl|=.|#.|\%[0-9][0-9]|\w|\W)'
        # smi_pat = r'(\[.*?\]|Br|Cl|\%[0-9][0-9]|\w|\W)'
        smi_pat = r'(\[.*?\]|Br|Cl|(?<=%)[0-9][0-9]|\w|\W)'

        # smi_list = list(filter(None, re.split(smi_pat, smiles)))
        smi_list = list(filter(lambda x: not ((x == "") or (x == "%")), re.split(smi_pat, smiles)))

        # combine bond with next token only if the next token isn't a number
        # assume SMILES does not end with a bonding character!
        tmp_idx = [i for i, x in enumerate(smi_list) if ((x in "-=#") and (not smi_list[i + 1].isdigit()))]
        if len(tmp_idx) > 0:
            for i in tmp_idx:
                smi_list[i + 1] = smi_list[i] + smi_list[i + 1]
            smi_list = np.delete(smi_list, tmp_idx).tolist()

        return smi_list

    @classmethod
    def smi2esmi(cls, smi):
        smi_list = cls.smi2list(smi)

        esmi_list = smi_list + ['!']
        substr_list = []  # list of all contracted substrings (include current char.)

        # list of whether open branch exist at current character position (include current char.)
        br_list = []

        # list of number of open ring at current character position (include current char.)
        ring_list = []
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
            elif esmi_list[i].isdigit():
                esmi_list[i] = int(esmi_list[i])
                if esmi_list[i] in v_ringn:
                    esmi_list[i] = v_ringn.index(esmi_list[i])
                    v_ringn.pop(esmi_list[i])
                else:
                    v_ringn.insert(0, esmi_list[i])
                    esmi_list[i] = '&'

            tmp_ss.append(esmi_list[i])
            substr_list.append(deepcopy(tmp_ss))
            br_list.append(n_br)
            ring_list.append(len(v_ringn))

        return pd.DataFrame({
            'esmi': esmi_list,
            'n_br': br_list,
            'n_ring': ring_list,
            'substr': substr_list
        })

    # may add error check here in the future?
    @classmethod
    def esmi2smi(cls, ext_smi):
        smi_list = ext_smi['esmi'].tolist()
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
        return ''.join(smi_list)

    def remove_table(self, max_order = None):
        """
        Remove estimators from estimator set.

        Parameters
        ----------
        max_order: int
            max order to be left in the table, the rest is removed.
        """
        if max_order:
            tmp = self._train_order[1] - max_order
            if tmp < 1:
                warnings.warn('Nothing removed', RuntimeWarning)
            else:
                self._table = self._table[:-tmp]
                self._train_order = (self._train_order[0], max_order)
        else:
            self._table = None
            self._train_order = None

    def fit(self, smiles, *, train_order=(1, 10)):
        """

        Parameters
        ----------
        smiles: list[str]
            SMILES for training.
        train_order: tuple[int, int] or int
            range of order when train a NGram table,
            when given int, train_order = (1, int),
            and train_order[0] must be > 0

        Returns
        -------

        """

        def _fit_one(ext_smi):
            for iB in [0, 1]:
                # index for open/closed branches char. position, remove last row for '!'
                idx_B = ext_smi.iloc[:-1].index[(ext_smi['n_br'].iloc[:-1] > 0) == iB]
                list_R = ext_smi['n_ring'][idx_B].unique().tolist()
                if len(list_R) > 0:
                    # expand list of dataframe for max. num-of-ring + 1
                    if len(self._table[0][iB]) < (max(list_R) + 1):
                        for ii in range(len(self._table)):
                            self._table[ii][iB].extend([
                                pd.DataFrame()
                                for i in range((max(list_R) + 1) - len(self._table[ii][iB]))
                            ])
                    for iR in list_R:
                        # index for num-of-open-ring char. pos.
                        idx_R = idx_B[ext_smi['n_ring'][idx_B] == iR]

                        # shift one down for 'next character given substring'
                        tar_char = ext_smi['esmi'][idx_R + 1].tolist()
                        tar_substr = ext_smi['substr'][idx_R].tolist()

                        for iO in range(self._train_order[0]-1, self._train_order[1]):
                            # index for char with substring length not less than order
                            idx_O = [x for x in range(len(tar_substr)) if len(tar_substr[x]) > iO]
                            for iC in idx_O:
                                if not tar_char[iC] in self._table[iO][iB][iR].columns.tolist():
                                    self._table[iO][iB][iR][tar_char[iC]] = 0
                                tmp_row = str(tar_substr[iC][-(iO + 1):])
                                if tmp_row not in self._table[iO][iB][iR].index.tolist():
                                    self._table[iO][iB][iR].loc[tmp_row] = 0

                                # somehow 'at' not ok with mixed char and int column names
                                self._table[iO][iB][iR].loc[tmp_row, tar_char[iC]] += 1

        if self._table:
            raise RuntimeError('NGram table existed.'
                               'If you want to re-train the table,'
                               'please use `remove_table()` method first.')

        if isinstance(train_order, int):
            tmp_train_order = (1, train_order)
        elif isinstance(train_order, tuple):
            tmp_train_order = train_order
        elif isinstance(train_order, (list,np.array,pd.Series)):
            tmp_train_order = (train_order[0],train_order[1])
        else:
            raise TypeError('please input a <tuple> of two <int> or a single <int> for train_order')

        if tmp_train_order[0] < 1:
            raise RuntimeError('Min train_order must be greater than 0')
        if tmp_train_order[1] < tmp_train_order[0]:
            raise RuntimeError('Min train_order must not be smaller than max train_order')

        self._train_order = tmp_train_order
        self._table = [[[], []] for _ in range(self._train_order[1])]

        self._fit_sample_order()
        self._fit_min_len()
        for smi in tqdm(smiles):
            try:
                _fit_one(self.smi2esmi(smi))
            except Exception as e:
                warnings.warn('NGram training failed for %s' % smi, RuntimeWarning)
                e = NGramTrainingError(e, smi)
                self.on_errors(e)

        return self

    # get probability vector for sampling next character, return character list and corresponding probability in numpy.array (normalized)
    # may cause error if empty string list is fed into 'tmp_str'
    # Warning: maybe can reduce the input of iB and iR - directly input the reduced list of self._ngram_tab (?)
    # Warning: may need to update this function with bisection search for faster speed (?)
    # Warning: may need to add worst case that no pattern found at all?

    def get_prob(self, tmp_str, iB, iR):
        # right now we use back-off method, an alternative is Kneser–Nay smoothing
        cand_char = []
        cand_prob = 1
        iB = int(iB)
        for iO in range(self.sample_order[1] - 1, self.sample_order[0] - 2, -1):
            # if (len(tmp_str) > iO) & (str(tmp_str[-(iO + 1):]) in self._table[iO][iB][iR].index.tolist()):
            if len(tmp_str) > iO and str(tmp_str[-(iO + 1):]) in self._table[iO][iB][iR].index.tolist():
                cand_char = self._table[iO][iB][iR].columns.tolist()
                cand_prob = np.array(self._table[iO][iB][iR].loc[str(tmp_str[-(iO + 1):])])
                break
        if len(cand_char) == 0:
            warnings.warn('get_prob: %s not found in NGram, iB=%i, iR=%i' % (tmp_str, iB, iR), RuntimeWarning)
            raise GetProbError(tmp_str, iB, iR)
        return cand_char, cand_prob / np.sum(cand_prob)

    # get the next character, return the probability value
    def sample_next_char(self, ext_smi):
        iB = ext_smi['n_br'].iloc[-1] > 0
        iR = ext_smi['n_ring'].iloc[-1]
        cand_char, cand_prob = self.get_prob(ext_smi['substr'].iloc[-1], iB, iR)
        # here we assume cand_char is not empty
        idx = np.random.choice(range(len(cand_char)), p=cand_prob)
        next_char = cand_char[idx]
        ext_smi = self.add_char(ext_smi, next_char)
        return ext_smi, cand_prob[idx]

    @classmethod
    def add_char(cls, ext_smi, next_char):
        new_pd_row = ext_smi.iloc[-1]
        new_pd_row.at['substr'] = new_pd_row['substr'] + [next_char]
        new_pd_row.at['esmi'] = next_char
        if next_char == '(':
            new_pd_row.at['n_br'] += 1
        elif next_char == ')':
            new_pd_row.at['n_br'] -= 1
            # assume '(' must exist before if the extended SMILES is valid! (will fail if violated)
            # idx = next((x for x in range(len(new_pd_row['substr'])-1,-1,-1) if new_pd_row['substr'][x] == '('), None)
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
        return ext_smi.append(new_pd_row, ignore_index=True)

    @classmethod
    def del_char(cls, ext_smi, n_char):
        if n_char > 0:
            return ext_smi[:-n_char]
        else:
            return ext_smi

    # need to make sure esmi_pd is a completed SMILES to use this function
    # todo: kekuleSmiles?
    @classmethod
    def reorder_esmi(cls, ext_smi):
        # convert back to SMILES first, then to rdkit MOL
        mol = Chem.MolFromSmiles(cls.esmi2smi(ext_smi))
        idx = np.random.choice(range(mol.GetNumAtoms())).item()
        # currently assume kekuleSmiles=True, i.e., no small letters but with ':' for aromatic rings
        ext_smi = cls.smi2esmi(Chem.MolToSmiles(mol, rootedAtAtom=idx))

        return ext_smi

    def validator(self, ext_smi):
        # delete all ending '(' or '&'
        for i in range(len(ext_smi)):
            if not ((ext_smi['esmi'].iloc[-1] == '(') or (ext_smi['esmi'].iloc[-1] == '&')):
                break
            ext_smi = self.del_char(ext_smi, 1)
        # delete or fill in ring closing
        flag_ring = ext_smi['n_ring'].iloc[-1] > 0
        for i in range(len(ext_smi)):  # max to double the length of current SMILES
            if flag_ring and (np.random.random() < 0.7):  # 50/50 for adding two new char.
                # add a character
                ext_smi, _ = self.sample_next_char(ext_smi)
                flag_ring = ext_smi['n_ring'].iloc[-1] > 0
            else:
                break
        if flag_ring:
            # prepare for delete (1st letter shall not be '&')
            tmp_idx = ext_smi.iloc[1:].index
            tmp_count = np.array(ext_smi['n_ring'].iloc[1:]) - np.array(ext_smi['n_ring'].iloc[:-1])
            num_open = tmp_idx[tmp_count == 1].values.tolist()
            num_open.reverse()
            num_close = tmp_idx[tmp_count == -1].values.tolist()
            idx_pop = []
            for i in num_close:
                idx_pop.append(ext_smi['esmi'][i])
            for ii, i in enumerate(idx_pop):
                ext_smi['esmi'][num_close[ii]] += sum([x < i for x in idx_pop[ii+1:]]) - i 
                num_open.pop(i)
            # delete all irrelevant rows and reconstruct esmi
            ext_smi = self.smi2esmi(
                self.esmi2smi(ext_smi.drop(ext_smi.index[num_open]).reset_index(drop=True)))
            ext_smi = ext_smi.iloc[:-1]  # remove the '!'

            # delete ':' that are not inside a ring
        # tmp_idx = esmi_pd.index[(esmi_pd['esmi'] == ':') & (esmi_pd['n_ring'] < 1)]
        # if len(tmp_idx) > 0:
        #     esmi_pd = smi2esmi(esmi2smi(esmi_pd.drop(tmp_idx).reset_index(drop=True)))
        #     esmi_pd = esmi_pd.iloc[:-1] # remove the '!'
        # fill in branch closing (last letter shall not be '(')
        for i in range(ext_smi['n_br'].iloc[-1]):
            ext_smi = self.add_char(ext_smi, ')')

        return ext_smi

    def proposal(self, smiles):
        """
        Propose new SMILES based on the given SMILES.
        Make sure you always check the train_order against sample_order before using the proposal!

        Parameters
        ----------
        smiles: list of SMILES
            Given SMILES for modification.

        Returns
        -------
        new_smiles: list of SMILES
            The proposed SMILES from the given SMILES.
        """
        new_smis = []
        for i, smi in enumerate(smiles):
            ext_smi = self.smi2esmi(smi)
            try:
                new_ext_smi = self.modify(ext_smi)
                new_smi = self.esmi2smi(new_ext_smi)
                if Chem.MolFromSmiles(new_smi) is None:
                    warnings.warn('can not convert %s to Mol' % new_smi, RuntimeWarning)
                    raise MolConvertError(new_smi)
                new_smis.append(new_smi)

            except ProposalError as e:
                e.old_smi = smi
                new_smi = self.on_errors(e)
                new_smis.append(new_smi)

            except Exception as e:
                raise e

        return new_smis

    def _merge_table(self, ngram_tab, weight=1):
        """
        Merge with a given NGram table

        Parameters
        ----------
        ngram_tab: NGram
            the table in the given NGram class variable will be merged to the table in self
        weight: double
            a scalar to scale the frequency in the given NGram table

        Returns
        -------
        tmp_n_gram: NGram
            merged NGram tables
        """

        self._train_order = (min(self._train_order[0],ngram_tab._train_order[0]), max(self._train_order[1],ngram_tab._train_order[1]))
        self._fit_sample_order()
        self._fit_min_len()

        n_gram_tab1 = self._table  # do not use deepcopy here
        n_gram_tab2 = ngram_tab.ngram_table  # default deepcopy used
        w = weight

        ord1 = len(n_gram_tab1)
        ord2 = len(n_gram_tab2)
        Bc1 = len(n_gram_tab1[0][0])
        Bc2 = len(n_gram_tab2[0][0])
        Bo1 = len(n_gram_tab1[0][1])
        Bo2 = len(n_gram_tab2[0][1])

        # fix the number of ring mis-match first
        if Bc1 < Bc2:
            for ii in range(ord1):
                n_gram_tab1[ii][0].extend([
                    pd.DataFrame()
                    for _ in range(Bc2 - Bc1)
                ])
        elif Bc1 > Bc2:
            for ii in range(ord2):
                n_gram_tab2[ii][0].extend([
                    pd.DataFrame()
                    for _ in range(Bc1 - Bc2)
                ])
        if Bo1 < Bo2:
            for ii in range(ord1):
                n_gram_tab1[ii][1].extend([
                    pd.DataFrame()
                    for _ in range(Bo2 - Bo1)
                ])
        elif Bo1 > Bo2:
            for ii in range(ord2):
                n_gram_tab2[ii][1].extend([
                    pd.DataFrame()
                    for _ in range(Bo1 - Bo2)
                ])

        # fix order mis-match
        if ord2 > ord1:
            n_gram_tab1.extend(n_gram_tab2[ord1:])

        # combine overlapped order (weighted on tab2)
        for i in range(min(ord1, ord2)):
            for j in range(len(n_gram_tab1[i])):
                for k in range(len(n_gram_tab1[i][j])):
                    n_gram_tab1[i][j][k] = n_gram_tab1[i][j][k].add(w * n_gram_tab2[i][j][k], fill_value=0).fillna(0)

    def merge_table(self, *ngram_tab: 'NGram', weight=1, overwrite=True):
        """
        Merge with a given NGram table

        Parameters
        ----------
        ngram_tab
            the table(s) in the given NGram class variable(s) will be merged to the table in self
        weight: int/float or list/tuple/np.array/pd.Series[int/float]
            a scalar/vector to scale the frequency in the given NGram table to be merged,
            must have the same length as ngram_tab
        overwrite: boolean
            overwrite the original table (self) or not,
            do not recommend to be False (may have memory issue)

        Returns
        -------
        tmp_n_gram: NGram
            merged NGram tables
        """

        if not np.all([isinstance(x, NGram) for x in ngram_tab]):
            raise TypeError('each element in the input must be <NGram>')

        if isinstance(weight, (int, float)):
            weight = np.repeat(weight, len(ngram_tab))
        elif isinstance(weight, (tuple, list, np.array, pd.Series)):
            if not np.all([isinstance(x, (int, float)) for x in weight]):
                raise TypeError('each element in weight must be <int> or <float>')
        else:
            raise TypeError('weight must be <int> or <float> or a list of them')

        if overwrite:
            tmp_n_gram = self  # do not use deepcopy here
        else:
            tmp_n_gram = deepcopy(self)

        for i, tab in enumerate(ngram_tab):
            tmp_n_gram._merge_table(ngram_tab=tab, weight=weight[i])

        return tmp_n_gram

    def split_table(self, cut_order):
        """
        Split NGram table into two

        Parameters
        ----------
        cut_order: int
            split NGram table between cut_order and cut_order+1

        Returns
        -------
        n_gram1: NGram
        n_gram2: NGram
        """

        n_gram1 = deepcopy(self)
        n_gram1.remove_table(max_order=cut_order)
        n_gram1._fit_sample_order()
        n_gram1._fit_min_len()

        n_gram2 = deepcopy(self)
        for iB in [0, 1]:
            for ii in range(cut_order):
                n_gram2._table[ii][iB] = [pd.DataFrame() for _ in range(len(n_gram2._table[ii][iB]))]
        n_gram2._train_order = (cut_order+1, self._train_order[1])
        n_gram2._fit_sample_order()
        n_gram2._fit_min_len()

        return n_gram1, n_gram2
