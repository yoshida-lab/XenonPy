#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import warnings

import numpy as np
import torch
from pymatgen.core.structure import Structure

from xenonpy.datatools import preset
from xenonpy.descriptor.base import BaseFeaturizer

__all__ = ['CrystalGraphFeaturizer']


class CrystalGraphFeaturizer(BaseFeaturizer):

    def __init__(self, *, max_num_nbr=12, radius=8, atom_feature='origin', n_jobs=-1, on_errors='raise',
                 return_type='any'):
        """
        This featurizer is a port of the original paper [CGCNN]_.

        .. [CGCNN] `Crystal Graph Convolutional Neural Networks for an Accurate and
            Interpretable Prediction of Material Properties`__
        __ https://doi.org/10.1103/PhysRevLett.120.145301

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
        self.atom_feature = atom_feature
        self.radius = radius
        self.max_num_nbr = max_num_nbr
        self.__implement__ = ['TsumiNa']

    def _atom_feature(self, atom_symbol: str):
        if self.atom_feature == 'origin':
            return preset.atom_init.loc[atom_symbol]
        elif self.atom_feature == 'elements':
            return preset.elements_completed.loc[atom_symbol]
        elif callable(self.atom_feature):
            return self.atom_feature(atom_symbol)
        else:
            raise TypeError('bad `atom feature` parameter')

    def edge_features(self, structure: Structure, **kwargs):
        def expand_distance(distances, dmin=0, step=0.2, var=None):
            """
            Parameters
            ----------
    
            dmin: float
              Minimum interatomic distance
            dmax: float
              Maximum interatomic distance
            step: float
              Step size for the Gaussian filter
            """
            filter_ = np.arange(dmin, self.radius + step, step)
            if var is None:
                var = step

            return np.exp(-(distances[..., np.newaxis] - filter_) ** 2 / var ** 2)

        all_nbrs = structure.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea = [], []
        for nbr in all_nbrs:
            if len(nbr) < self.max_num_nbr:
                warnings.warn('can not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.')
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
        nbr_fea = np.array(nbr_fea)
        nbr_fea = expand_distance(nbr_fea)
        nbr_fea = torch.Tensor(nbr_fea)

        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)

        return nbr_fea, nbr_fea_idx

    def node_features(self, structure: Structure):
        atom_features = np.vstack([self._atom_feature(s.name) for s in structure.species])
        return torch.Tensor(atom_features)

    def featurize(self, structure: Structure):
        return [self.node_features(structure), *self.edge_features(structure)]

    @property
    def feature_labels(self):
        return ['atom_feature', 'neighbor_feature', 'neighbor_idx']
