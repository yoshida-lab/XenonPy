#  Copyright (c) 2019. TsumiNa. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from .base import BaseGraphFeaturizer


class CrystalGraphFeaturizer(BaseGraphFeaturizer):

    def __init__(self, *, n_jobs=-1, on_errors='raise', return_type='any'):
        """

        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type: str
            Specific the return type.
            Can be ``any``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``numpy.ndarray`` and ``pandas.DataFrame`` respectively.
            If ``any``, the return type dependent on the input type.
            Default is ``any``
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type)

    def edge_features(self, *x, **kwargs):
        pass

    def node_features(self, *x, **kwargs):
        pass
