# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin


# from ..decorator import combinator


class PDFDesc(BaseEstimator, TransformerMixin):
    """
    Calculate pair distribution descriptor for ML.

    This script based on
    `pymatgen-diffusion <https://github.com/materialsvirtuallab/pymatgen-diffusion>`_
    with some modify.
    """

    def __init__(self,
                 n_grid: int = 151,
                 r_max: float = 15.0,
                 *,
                 cell_range: int = 1,
                 sigma: float = 0.1,
                 structure_col: str = 'structure',
                 verbose: bool = True,
                 ):
        """

        Parameters
        ----------
        n_grid: int
            Number of radial grid points.
        r_max: float
            Maximum of radial grid (the minimum is always set zero).
        cell_range: int
            Range of translational vector elements associated
            with supercell. Default is 1, i.e. including the adjacent image
            cells along all three directions.
        sigma: float
            Smearing of a Gaussian function.
        structure_col: str
            Column's name contain structure objects.
        verbose: bool
            If verbose output.
        """

        assert n_grid >= 1, "ngrid should be greater than 1!"
        assert sigma > 0, "sigma should be a positive number!"

        self.structures = list()
        self.structure_col = structure_col

        self.dr = r_max / (n_grid - 1)
        self.verbose = verbose
        self.pdfs = None
        self.dns = Counter()
        self.interval = np.linspace(0.0, r_max, n_grid)
        self.rmax = r_max
        self.ngrid = n_grid
        self.sigma = sigma

        # generate the translational vectors
        r = np.arange(cell_range, cell_range + 1)
        arange = r[:, None] * np.array([1, 0, 0])[None, :]
        brange = r[:, None] * np.array([0, 1, 0])[None, :]
        crange = r[:, None] * np.array([0, 0, 1])[None, :]
        images = arange[:, None, None] + brange[None, :, None] + crange[None, None, :]
        self.images = images.reshape((len(r) ** 3, 3))
        self.r_arange = r

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
            ret = pd.DataFrame(np.array(ret), index=structures.index, columns=self.interval)

            end = time.time()
            print('using {:.5f}s'.format(end - start))
        else:
            ret = structures[self.structure_col].apply(self.pdf).tolist()
            ret = pd.DataFrame(np.array(ret), index=structures.index, columns=self.interval)

        self.pdfs = ret
        return ret

    def pdf(self, structure):
        """
        calculate pair distribution from a pymatgen Structure object.

        :param Structure structure: Structure object contain atoms coordination
        :return: self
        :rtype: self
        """
        lattice = structure.lattice
        species = structure.symbol_set

        indices = [
            j for j, site in enumerate(structure)
            if site.specie.symbol in species
        ]

        assert indices, "Given species are not in the structure!"
        ref_indices = indices

        rho = float(len(indices)) / lattice.volume
        all_fcoords = np.array(structure.frac_coords)
        fcoords_list = [all_fcoords[indices, :]]
        ref_fcoords_list = [all_fcoords[ref_indices, :]]

        interval = np.linspace(0.0, self.rmax, self.ngrid)
        pdf = np.zeros(self.ngrid, dtype=np.double)
        dns = Counter()

        # find the zero image vector
        zd = np.sum(self.images ** 2, axis=1)
        indx0 = np.argmin(zd)

        for fcoords, ref_fcoords in zip(fcoords_list, ref_fcoords_list):
            dcf = fcoords[:, None, None, :] + self.images[None, None, :, :] - ref_fcoords[None, :, None, :]
            dcc = lattice.get_cartesian_coords(dcf)
            d2 = np.sum(dcc ** 2, axis=3)
            dists = [
                d2[u, v, j] ** 0.5
                for u in range(len(indices)) for v in range(len(ref_indices))
                for j in range(len(self.r_arange) ** 3) if u != v or j != indx0
            ]
            dists = filter(lambda e: e < self.rmax + 1e-8, dists)
            r_indices = [int(dist / self.dr) for dist in dists]
            dns.update(r_indices)

        for indx, dn in dns.most_common(self.ngrid):
            if indx > len(interval) - 1:
                continue

            if indx == 0:
                ff = np.pi * self.dr ** 2
            else:
                ff = 4.0 * np.pi * interval[indx] ** 2

            pdf[:] += stats.norm.pdf(
                interval, interval[indx], self.sigma) * dn / float(
                len(ref_indices)) / ff / rho / len(fcoords_list)

        self.structures.append(structure)
        return pdf

    def to_csv(self, file: str):
        """
        Output RDF data to a csv file.

        :param str file: file path to save.
        """
        self.pdfs.to_csv(file)
