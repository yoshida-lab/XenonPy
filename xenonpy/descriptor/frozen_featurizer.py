#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import warnings

import numpy as np
import pandas as pd
import torch as tc

from .base import BaseFeaturizer


class FrozenFeaturizer(BaseFeaturizer):
    """
    A Featurizer to extract hidden layers a from NN model.
    """

    def __init__(self, model=None, *, cuda=False, depth=None, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        model: torch.nn.Module
            Source model.
        cuda: bool
            If ``true``, run on GPU.
        depth: int
            The depth will be retrieved from NN model.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """
        super().__init__(n_jobs=0, on_errors=on_errors, return_type=return_type)
        self.depth = depth
        self.model = model
        self.cuda = cuda
        self._ret = []

    def featurize(self, descriptor, *, depth=None):
        if not isinstance(self.model, tc.nn.Module):
            raise TypeError('<model> must be a instance of <torch.nn.Module>')
        hlayers = []
        if isinstance(descriptor, pd.DataFrame):
            descriptor = descriptor.values
        x_ = tc.from_numpy(descriptor).type(tc.FloatTensor)
        if self.cuda:
            x_.cuda()
            self.model.cuda()
        else:
            x_.cpu()
            self.model.cpu()
        for l in self.model[:-1]:
            x_ = l.layer(x_)
            hlayers.append(x_.data)

        if depth is None:
            depth = self.depth
        if depth is not None:
            ret = hlayers[-depth:]
            if depth > len(ret):
                warnings.warn('<depth> is greater than the max depth of hidden layers')
        else:
            ret = hlayers
            self.depth = len(ret)
        if self.cuda:
            ret = [l.cpu().numpy() for l in ret]
        else:
            ret = [l.numpy() for l in ret]
        self._ret = ret
        return np.concatenate(ret, axis=1)

    @property
    def feature_labels(self):
        depth = len(self._ret)
        if depth == 0:
            raise ValueError('Can not generate labels before transform.')
        return ['L(' + str(i - depth) + ')_' + str(j + 1)
                for i in range(len(self._ret))
                for j in range(self._ret[i].shape[1])]
