#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from collections import OrderedDict
from typing import Union, Dict, Callable, Sequence, Any, Optional

import numpy as np
import pandas as pd

__all__ = ['ParameterGenerator']


class ParameterGenerator(object):
    """
    Generator for parameter set generating.

    """

    def __init__(self, seed: Optional[int] = None, **kwargs: Union[Any, Sequence, Callable, Dict]):
        """

        Parameters
        ----------
        seed
            Numpy random seed.
        kwargs
            Parameter candidate.
        """
        if len(kwargs) == 0:
            raise RuntimeError('need parameter candidate')

        np.random.seed(seed)

        self.tuples = OrderedDict()
        self.funcs = OrderedDict()
        self.dicts = OrderedDict()
        self.others = {}

        for k, v in kwargs.items():
            if isinstance(v, (tuple, list, np.ndarray, pd.Series)):
                self.tuples[k] = v
            elif callable(v):
                self.funcs[k] = v
            elif isinstance(v, dict):
                repeat = v['repeat']
                self.dicts[k] = v

                if isinstance(repeat, str):
                    if repeat in self.tuples:
                        self.tuples.move_to_end(repeat, True)
                    if repeat in self.dicts:
                        self.dicts.move_to_end(repeat, True)
                    if repeat in self.funcs:
                        self.funcs.move_to_end(repeat, True)
            else:
                self.others[k] = v

    def __call__(self, num: int, *, factory=None):
        for _ in range(num):
            tmp = {}
            for k, v in self.tuples.items():
                tmp[k] = self._gen(v)

            for k, v in self.funcs.items():
                tmp[k] = v()

            for k, v in reversed(self.dicts.items()):
                data = v['data']
                repeat = v['repeat']
                if 'replace' in v:
                    replace = v['replace']
                else:
                    replace = True

                if isinstance(repeat, (tuple, list, np.ndarray, pd.Series)):
                    repeat = self._gen(repeat)
                elif isinstance(repeat, str):
                    repeat = len(tmp[repeat])

                if isinstance(data, (tuple, list, np.ndarray, pd.Series)):
                    tmp[k] = self._gen(data, repeat, replace)
                elif callable(data):
                    tmp[k] = tuple(data(repeat))

            tmp = dict(self.others, **tmp)
            if factory is not None:
                yield tmp, factory(**tmp)
            else:
                yield tmp

    @staticmethod
    def _gen(item: Sequence, repeat: int = None, replace: bool = True):
        if repeat is not None:
            idx = np.random.choice(len(item), repeat, replace=replace)
            return tuple([item[i] for i in idx])
        else:
            idx = np.random.choice(len(item))
            return item[idx]
