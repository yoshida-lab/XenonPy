# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from matminer.featurizers.structure import RadialDistributionFunction
from sklearn.base import BaseEstimator, TransformerMixin

# from ..pipeline import combinator


class PDFDesc(BaseEstimator, TransformerMixin):
    """
    Calculate pair distribution descriptor for machine learning.

    """

    def __init__(self,
                 n_grid: int = 151,
                 r_max: float = 20.0,
                 *,
                 structure_col: str = 'structure',
                 verbose: bool = True):
        """

        Parameters
        ----------
        n_grid: int
            Number of radial grid points.
        r_max: float
            Maximum of radial grid (the minimum is always set zero).
        structure_col: str
            Column's name contain structure objects.
        verbose: bool
            If verbose output.
        """

        assert n_grid >= 1, "ngrid should be greater than 1!"

        self.structures = list()
        self.structure_col = structure_col
        self.r_max = r_max
        self.dr = r_max / (n_grid - 1)
        self.interval = np.arange(0, r_max + self.dr, self.dr)
        self.rdf = RadialDistributionFunction(r_max, self.dr)
        self.verbose = verbose
        self.pdfs = None

    def fit(self, structures):
        """

        Parameters
        ----------
        structures: Structure list
            pymatgen Structure object list.

        Returns
        -------
            self
        """
        if self.structure_col not in structures:
            raise ValueError('need structure to calculate descriptor')

        return self

    def transform(self, structures):
        if self.structure_col not in structures:
            raise ValueError('need structure to calculate descriptor')

        if self.verbose:
            import time
            start = time.time()
            print('converting {:d} structure'.format(structures.shape[0]))

            ret = structures[self.structure_col].apply(self.pdf).tolist()
            ret = pd.DataFrame(
                np.array(ret), index=structures.index, columns=self.interval)

            end = time.time()
            print('using {:.5f}s'.format(end - start))
        else:
            ret = structures[self.structure_col].apply(self.pdf).tolist()
            ret = pd.DataFrame(
                np.array(ret), index=structures.index, columns=self.interval)

        self.pdfs = ret
        return ret

    def _pdf(self, s):
        _, rdf = self.rdf(s)[0]
        pass

    def to_csv(self, file: str):
        """
        Output RDF data to a csv file.

        :param str file: file path to save.
        """
        self.pdfs.to_csv(file)
