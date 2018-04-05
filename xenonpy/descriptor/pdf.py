# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
import pandas as pd
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
        self.interval = np.arange(self.dr, r_max + self.dr, self.dr)
        self.verbose = verbose

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

        return ret

    def pdf(self, s):
        """
        Get RDF of the input structure.
        Args:
            s: Pymatgen Structure object.
        Returns:
            rdf, dist: (tuple of arrays) the first element is the
                    normalized RDF, whereas the second element is
                    the inner radius of the RDF bin.
        """
        if not s.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get the distances between all atoms
        neighbors_lst = s.get_all_neighbors(self.r_max)
        all_distances = np.concatenate(
            tuple(map(lambda x: [itemgetter(1)(e) for e in x], neighbors_lst)))

        # Compute a histogram
        dist_hist, dist_bins = np.histogram(
            all_distances, bins=np.arange(
                0, self.r_max + self.dr, self.dr), density=False)

        # Normalize counts
        shell_vol = 4.0 / 3.0 * np.pi * (np.power(
            dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        number_density = s.num_sites / s.volume
        return dist_hist / shell_vol / number_density

    def to_csv(self, file: str):
        """
        Output RDF data to a csv file.

        :param str file: file path to save.
        """
        self.pdfs.to_csv(file)
