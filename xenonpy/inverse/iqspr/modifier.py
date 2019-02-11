#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import pandas as pd

from .base import BaseModifier


class NGram(BaseModifier):
    def __init__(self):
        self.n_gram_table = None

    def fit(self, ext_smi, **kwargs):
        for iB in [False, True]:
            # index for open/closed branches char. position, remove last row for '!'
            idx_B = ext_smi.iloc[:-1].index[(ext_smi['n_br'].iloc[:-1] > 0) == iB]
            list_R = ext_smi['n_ring'][idx_B].unique().tolist()
            if len(list_R) > 0:
                if len(self._modifier[0][iB]) < (
                        max(list_R) + 1):  # expand list of dataframe for max. num-of-ring + 1
                    for ii in range(len(self._modifier)):
                        self._modifier[ii][iB].extend([
                            pd.DataFrame()
                            for i in range((max(list_R) + 1) - len(self._modifier[ii][iB]))
                        ])
                for iR in list_R:
                    idx_R = idx_B[ext_smi['n_ring'][idx_B] ==
                                  iR]  # index for num-of-open-ring char. pos.
                    tar_char = ext_smi['esmi'][
                        idx_R + 1].tolist()  # shift one down for 'next character given substring'
                    tar_substr = ext_smi['substr'][idx_R].tolist()
                    for iO in range(len(self._modifier)):
                        idx_O = [x for x in range(len(tar_substr)) if len(tar_substr[x]) > iO
                                 ]  # index for char with substring length not less than order
                        for iC in idx_O:
                            if not tar_char[iC] in self._modifier[iO][iB][iR].columns.tolist():
                                self._modifier[iO][iB][iR][tar_char[iC]] = 0
                            tmp_row = str(tar_substr[iC][-(iO + 1):])
                            if not tmp_row in self._modifier[iO][iB][iR].index.tolist():
                                self._modifier[iO][iB][iR].loc[tmp_row] = 0
                            self._modifier[iO][iB][iR].loc[tmp_row, tar_char[
                                iC]] += 1  # somehow 'at' not ok with mixed char and int column names

        # return self._ngram_tab #maybe not needed?

    def transform(self, X):
        pass
