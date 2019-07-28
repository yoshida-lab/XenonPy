#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.


import numpy as np
from numpy import product


class Product(object):
    def __init__(self, *paras, repeat=1):
        if not isinstance(repeat, int):
            raise ValueError('repeat must be int but got {}'.format(
                type(repeat)))
        lens = [len(p) for p in paras]
        if repeat > 1:
            lens = lens * repeat
        size = product(lens)
        acc_list = [np.floor_divide(size, lens[0])]
        for len_ in lens[1:]:
            acc_list.append(np.floor_divide(acc_list[-1], len_))

        self.paras = paras * repeat if repeat > 1 else paras
        self.lens = lens
        self.size = size
        self.acc_list = acc_list

    def __getitem__(self, index):
        if index > self.size - 1:
            raise IndexError
        ret = [s - 1 for s in self.lens]  # from len to index
        remainder = index + 1
        for i, acc in enumerate(self.acc_list):
            quotient, remainder = np.divmod(remainder, acc)
            if remainder == 0:
                ret[i] = quotient - 1
                break
            ret[i] = quotient

        return tuple(self.paras[i][j] for i, j in enumerate(ret))

    def __len__(self):
        return self.size
