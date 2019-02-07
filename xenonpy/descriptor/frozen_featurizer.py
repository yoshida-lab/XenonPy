#  Copyright 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
import torch as tc

from .base import BaseFeaturizer


class FrozenFeaturizer(BaseFeaturizer):

    def __init__(self, model=None, cuda=False, *, on_errors='raise'):
        """
        Base class for composition feature.
        """
        super().__init__(n_jobs=0, on_errors=on_errors)
        self._model = model
        self._cuda = cuda
        self._ret = []
        self._depth = None

    def featurize(self, descriptor, depth=None):
        hlayers = []
        if isinstance(descriptor, pd.DataFrame):
            descriptor = descriptor.values
        x_ = tc.from_numpy(descriptor).type(tc.FloatTensor)
        if self._cuda:
            x_.cuda()
            self._model.cuda()
        else:
            x_.cpu()
            self._model.cpu()
        for l in self._model:
            x_ = l.layer(x_)
            hlayers.append(x_.data)

        if depth is not None:
            ret = hlayers[-depth - 1:-1]
            self._depth = depth
        else:
            ret = hlayers[:-1]
            self._depth = len(ret)
        if self._cuda:
            ret = [l.cpu().numpy() for l in ret]
        else:
            ret = [l.numpy() for l in ret]
        self._ret = ret
        return np.concatenate(ret, axis=1)

    @property
    def feature_labels(self):
        if len(self._ret) == 0:
            raise ValueError('Can not generate labels before transform.')
        return ['L(' + str(i - self._depth) + ')_' + str(j + 1)
                for i in range(len(self._ret))
                for j in range(self._ret[i].shape[1])]
