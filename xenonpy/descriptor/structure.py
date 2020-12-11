#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import re

import numpy as np
import pymatgen as pm
from pymatgen.analysis.local_env import VoronoiNN

from xenonpy.descriptor.base import BaseDescriptor, BaseFeaturizer

__all__ = ['RadialDistributionFunction', 'OrbitalFieldMatrix', 'Structures']


class RadialDistributionFunction(BaseFeaturizer):
    """
    Calculate pair distribution descriptor for machine learning.

    """

    @property
    def feature_labels(self):
        return [str(d) for d in self._interval[1:]]

    def __init__(self, n_bins=201, r_max=20.0, *, n_jobs=-1, on_errors='raise', return_type='any', target_col=None):
        """
        
        Parameters
        ----------
        n_bins: int
            Number of radial grid points.
        r_max: float
            Maximum of radial grid (the minimum is always set zero).
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """

        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        assert n_bins >= 1, "n_bins should be greater than 1!"
        assert r_max > 0, "r_max should be greater than 0!"

        self.n_bins = n_bins
        self.r_max = r_max
        self.dr = r_max / (n_bins - 1)
        self._interval = np.arange(0.0, r_max + self.dr, self.dr)
        self.__authors__ = ['TsumiNa']

    def featurize(self, structure):
        """
        Get RDF of the input structure.
        Args:
            structure: Pymatgen Structure object.
        Returns:
            rdf, dist: (tuple of arrays) the first element is the
                    normalized RDF, whereas the second element is
                    the inner radius of the RDF bin.
        """
        if not structure.is_ordered:
            raise ValueError("Disordered structure support not built yet")

        # Get the distances between all atoms
        neighbors_lst = structure.get_all_neighbors(self.r_max)
        all_distances = np.concatenate(tuple(map(lambda x: [e[1] for e in x], neighbors_lst)))

        # Compute a histogram
        dist_hist, dist_bins = np.histogram(all_distances, bins=self._interval, density=False)

        # Normalize counts
        shell_vol = 4.0 / 3.0 * np.pi * (np.power(dist_bins[1:], 3) - np.power(dist_bins[:-1], 3))
        number_density = structure.num_sites / structure.volume
        return dist_hist / shell_vol / number_density


class OrbitalFieldMatrix(BaseFeaturizer):
    """
    Representation based on the valence shell electrons of neighboring atoms.

    Each atom is described by a 32-element vector uniquely representing the
    valence subshell. A 32x32 (39x39) matrix is formed by multiplying two
    atomic vectors. An OFM for an atomic environment is the sum of these
    matrices for each atom the center atom coordinates with multiplied by a
    distance function (In this case, 1/r times the weight of the coordinating
    atom in the Voronoi.

    """

    def __init__(self, including_d=True, *, n_jobs=-1, on_errors='raise', return_type='any', target_col=None):
        """

        Parameters
        ----------
        including_d: bool
            If true, add distance information.
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
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
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(n_jobs=n_jobs, on_errors=on_errors, return_type=return_type, target_col=target_col)
        self._including_d = including_d
        self.__authors__ = ['TsumiNa']
        self.__citations__ = [
            '''
            @article{LamPham2017,
                archivePrefix = {arXiv},
                arxivId = {1705.01043},
                author = {{Lam Pham}, Tien and Kino, Hiori and Terakura, Kiyoyuki and Miyake, Takashi and Tsuda, Koji and Takigawa, Ichigaku and {Chi Dam}, Hieu},
                doi = {10.1080/14686996.2017.1378060},
                eprint = {1705.01043},
                issn = {18785514},
                journal = {Science and Technology of Advanced Materials},
                keywords = {Material descriptor,data mining,machine learning,magnetic materials,material informatics},
                number = {1},
                pages = {756--765},
                pmid = {29152012},
                publisher = {Taylor {\&} Francis},
                title = {{Machine learning reveals orbital interaction in materials}},
                url = {https://doi.org/10.1080/14686996.2017.1378060},
                volume = {18},
                year = {2017}
                }
            '''
        ]

    @staticmethod
    def get_element_representation(name):
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
            's1', 's2', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14'
        ]

        if name == 'H':
            element_electronic_structure = ['s1']
        elif name == 'He':
            element_electronic_structure = ['s2']
        else:
            element_electronic_structure = [
                ''.join(pair) for pair in re.findall(r"\.\d(\w+)<sup>(\d+)</sup>", element.electronic_structure)
            ]
        for eletron_subshell in element_electronic_structure:
            general_element_electronic[eletron_subshell] = 1.0

        return np.array([general_element_electronic[key] for key in general_electron_subshells])

    def featurize(self, structure, is_including_d=True):
        """
        Generate OFM descriptor

        Parameters
        ----------
        structure: pymatgen.Structure
            The input structure for OFM calculation.
        """

        atoms = np.array([site.species_string for site in structure])
        coordinator_finder = VoronoiNN(cutoff=10.0)

        local_orbital_field_matrices = []
        for i_atom, atom in enumerate(atoms):
            neighbors = coordinator_finder.get_nn_info(structure=structure, n=i_atom)

            site = structure[i_atom]
            center_vector = self.get_element_representation(atom)
            env_vector = np.zeros(32)

            for nn in neighbors:
                site_x = nn['site']
                w = nn['weight']
                site_x_label = site_x.species_string
                neigh_vector = self.get_element_representation(site_x_label)
                d = np.sqrt(np.sum((site.coords - site_x.coords)**2))
                if self._including_d:
                    env_vector += neigh_vector * w / d
                else:
                    env_vector += neigh_vector * w

            local_matrix = center_vector[None, :] * env_vector[:, None]
            local_matrix = np.ravel(local_matrix)
            local_orbital_field_matrices.append(local_matrix)

        return np.array(local_orbital_field_matrices).mean(axis=0)

    @property
    def feature_labels(self):
        labels = np.array([
            's1', 's2', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14'
        ])

        return [i + '_' + j for i in labels for j in labels]


class Structures(BaseDescriptor):
    """
    Calculate structure descriptors from compound's structure.
    """

    def __init__(self, n_bins=201, r_max=20.0, including_d=True,
                 *, n_jobs=-1, featurizers='all', on_errors='raise', target_col=None):
        """

        Parameters
        ----------
        n_bins: int
            Number of radial grid points.
        r_max: float
            Maximum of radial grid (the minimum is always set zero).
        including_d: bool
            If true, add distance information.
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
        featurizers: list[str] or 'all'
            Featurizers that will be used.
            Default is 'all'.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        """
        super().__init__(featurizers=featurizers)
        self.n_jobs = n_jobs

        self.structure = RadialDistributionFunction(n_bins, r_max, n_jobs=n_jobs,
                                                    on_errors=on_errors, target_col=target_col)
        self.structure = OrbitalFieldMatrix(including_d, n_jobs=n_jobs, on_errors=on_errors, target_col=target_col)
