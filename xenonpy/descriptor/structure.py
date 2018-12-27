# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

import re

import numpy as np
import pandas as pd
import pymatgen as pm
from pymatgen.analysis.local_env import VoronoiNN
from sklearn.base import BaseEstimator, TransformerMixin

from .base import BaseDescriptor


# from ..pipeline import combinator


class RadialDistributionFunction(BaseEstimator, TransformerMixin):
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
            ret = pd.DataFrame(np.array(ret), index=structures.index, columns=self.interval)

            end = time.time()
            print('using {:.5f}s'.format(end - start))
        else:
            ret = structures[self.structure_col].apply(self.pdf).tolist()
            ret = pd.DataFrame(np.array(ret), index=structures.index, columns=self.interval)

        return ret


class OrbitalFieldMatrix(BaseDescriptor):
    """
    Representation based on the valence shell electrons of neighboring atoms.

    Each atom is described by a 32-element vector uniquely representing the
    valence subshell. A 32x32 (39x39) matrix is formed by multiplying two
    atomic vectors. An OFM for an atomic environment is the sum of these
    matrices for each atom the center atom coordinates with multiplied by a
    distance function (In this case, 1/r times the weight of the coordinating
    atom in the Voronoi.
    
    """

    def __init__(self, n_jobs=-1):
        super().__init__()
        self._n_jobs = n_jobs

    def get_element_representation(self, name):
        """
        generate one-hot representation for a element, e.g, si = [0.0, 1.0, 0.0, 0.0, ...]

        Parameters
        ----------
        name: string
            element symbol
        """
        element = pm.Element(name)
        general_element_electronic = {
            's1': 0.0,
            's2': 0.0,
            'p1': 0.0,
            'p2': 0.0,
            'p3': 0.0,
            'p4': 0.0,
            'p5': 0.0,
            'p6': 0.0,
            'd1': 0.0,
            'd2': 0.0,
            'd3': 0.0,
            'd4': 0.0,
            'd5': 0.0,
            'd6': 0.0,
            'd7': 0.0,
            'd8': 0.0,
            'd9': 0.0,
            'd10': 0.0,
            'f1': 0.0,
            'f2': 0.0,
            'f3': 0.0,
            'f4': 0.0,
            'f5': 0.0,
            'f6': 0.0,
            'f7': 0.0,
            'f8': 0.0,
            'f9': 0.0,
            'f10': 0.0,
            'f11': 0.0,
            'f12': 0.0,
            'f13': 0.0,
            'f14': 0.0
        }

        general_electron_subshells = [
            's1', 's2', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
            'd7', 'd8', 'd9', 'd10', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10',
            'f11', 'f12', 'f13', 'f14'
        ]

        if name == 'H':
            element_electronic_structure = ['s1']
        elif name == 'He':
            element_electronic_structure = ['s2']
        else:
            element_electronic_structure = [
                ''.join(pair)
                for pair in re.findall("\.\d(\w+)<sup>(\d+)</sup>", element.electronic_structure)
            ]
        for eletron_subshell in element_electronic_structure:
            general_element_electronic[eletron_subshell] = 1.0

        return np.array([general_element_electronic[key] for key in general_electron_subshells])

    def struct2ofm(self, struct, is_ofm1=False, is_including_d=True):
        """
        Generate OFM descriptor
        """

        atoms = np.array([site.species_string for site in struct])
        coordinator_finder = VoronoiNN(cutoff=10.0)

        local_orbital_field_matrices = []
        for i_atom, atom in enumerate(atoms):
            neighbors = coordinator_finder.get_nn_info(structure=struct, n=i_atom)

            site = struct[i_atom]
            center_vector = self.get_element_representation(atom)
            env_vector = np.zeros(32)

            for nn in neighbors:
                site_x = nn['site']
                w = nn['weight']
                site_x_label = site_x.species_string
                neigh_vector = self.get_element_representation(site_x_label)
                d = np.sqrt(np.sum((site.coords - site_x.coords) ** 2))
                if is_including_d:
                    env_vector += neigh_vector * w / d
                else:
                    env_vector += neigh_vector * w

            local_matrix = center_vector[None, :] * env_vector[:, None]
            local_matrix = np.ravel(local_matrix)
            local_orbital_field_matrices.append(local_matrix)

        return np.array(local_orbital_field_matrices)
