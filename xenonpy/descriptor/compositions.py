#  Copyright (c) 2021. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

import itertools
from typing import Callable, List, Sequence, Union

import numpy as np
import pandas as pd
from pymatgen.core import Composition as PMGComp
from sklearn.preprocessing import MinMaxScaler
from xenonpy.datatools import preset
from xenonpy.descriptor.base import (BaseCompositionFeaturizer, BaseDescriptor, BaseFeaturizer)

__all__ = [
    'Compositions',
    'Counting',
    'WeightedAverage',
    'WeightedSum',
    'WeightedVariance',
    'HarmonicMean',
    'GeometricMean',
    'MaxPooling',
    'MinPooling',
    'KernelMean',
]


class KernelMean(BaseFeaturizer):

    def __init__(
        self,
        kernel_func: Union[None, Callable[[np.ndarray, np.ndarray], np.ndarray]],
        *,
        feature_matrix: Union[None, pd.DataFrame] = None,
        grid: Union[None, int, Sequence[int], Sequence[Sequence[float]]] = None,
    ):

        if feature_matrix is None:  # use elemental info
            feature_matrix = preset.elements_completed

        # re-scale to [0, 1]
        scaled_matrix = MinMaxScaler().fit_transform(feature_matrix)

        # calculate centers for each feature
        if grid is None:
            grid = scaled_matrix.values.mean(axis=0).reshape(-1, 1)  # use mean of feature as center
        elif isinstance(grid, int):
            grid = np.array([np.linspace(0, 1, grid)] * scaled_matrix.shape[1])  # create bins
        elif isinstance(grid, Sequence):
            grid = np.asarray(grid)
            if grid.ndim == 1:
                if grid.size != scaled_matrix.shape[1]:
                    raise ValueError(
                        f'length of grid ({grid.size}) must be equal to feature size ({scaled_matrix.shape[1]})')
                grid = np.array([np.linspace(0, 1, grid) for i in grid])
            elif grid.ndim == 2:
                pass  # direct input
            else:
                raise ValueError('dim of grid must be 1 or 2')

        # calculate kernel matrix for featrues
        kernel_matrix = kernel_func(scaled_matrix, grid)

        # generate column names of output
        labels = itertools.chain(
            *[[f'{n}_k{k+1}' for k in range(g.size)] for n, g in zip(feature_matrix.columns, grid)])

        self._kernel_matrix = pd.DataFrame(kernel_matrix, index=feature_matrix.index, columns=labels)

    def featurize(self, comp):
        if isinstance(comp, PMGComp):
            comp = comp.as_dict()

        return sum([self._kernel_matrix.loc[e].values for e, n in comp.items()])

    @property
    def feature_labels(self):
        return self._labels


class Counting(BaseCompositionFeaturizer):

    def __init__(self, *, one_hot_vec=False, n_jobs=-1, on_errors='raise', return_type='any', target_col=None):
        """

        Parameters
        ----------
        one_hot_vec : bool
            Set ``true`` to using one-hot-vector encoding.
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
        self.one_hot_vec = one_hot_vec
        self._elems = self.elements.index.tolist()
        self.__authors__ = ['TsumiNa']

    def mix_function(self, elems, nums):
        vec = np.zeros(len(self._elems), dtype=np.int)
        for i, e in enumerate(elems):
            if self.one_hot_vec:
                vec[self._elems.index(e)] = 1
            else:
                vec[self._elems.index(e)] = nums[i]

        return vec

    @property
    def feature_labels(self):
        return self._elems


class WeightedAverage(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
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
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, nums):
        elems_ = self.elements.loc[elems, :].values
        w_ = nums / np.sum(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['ave:' + s for s in self.elements]


class WeightedSum(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
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
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, nums):
        elems_ = self.elements.loc[elems, :].values
        w_ = np.array(nums)
        return w_.dot(elems_)

    @property
    def feature_labels(self):
        return ['sum:' + s for s in self.elements]


class GeometricMean(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
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
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, nums):
        elems_ = self.elements.loc[elems, :].values
        w_ = np.array(nums).reshape(-1, 1)
        tmp = elems_**w_
        return np.power(tmp.prod(axis=0), 1 / sum(w_))

    @property
    def feature_labels(self):
        return ['gmean:' + s for s in self.elements]


class HarmonicMean(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
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
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, nums):
        elems_ = 1 / self.elements.loc[elems, :].values
        w_ = np.array(nums)
        tmp = w_.dot(elems_)

        return sum(w_) / tmp

    @property
    def feature_labels(self):
        return ['hmean:' + s for s in self.elements]


class WeightedVariance(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
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
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, nums):
        elems_ = self.elements.loc[elems, :].values
        w_ = nums / np.sum(nums)
        mean_ = w_.dot(elems_)
        var_ = elems_ - mean_
        return w_.dot(var_**2)

    @property
    def feature_labels(self):
        return ['var:' + s for s in self.elements]


class MaxPooling(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
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
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, _):
        elems_ = self.elements.loc[elems, :]
        return elems_.max().values

    @property
    def feature_labels(self):
        return ['max:' + s for s in self.elements]


class MinPooling(BaseCompositionFeaturizer):
    """

    Parameters
    ----------
    elemental_info
        Elemental level information for each element. For example, the ``atomic number``,
        ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
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
        ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
        If ``any``, the return type dependent on the input type.
        Default is ``any``
    target_col
        Only relevant when input is pd.DataFrame, otherwise ignored.
        Specify a single column to be used for transformation.
        If ``None``, all columns of the pd.DataFrame is used.
        Default is None.
    """

    def mix_function(self, elems, _):
        elems_ = self.elements.loc[elems, :]
        return elems_.min().values

    @property
    def feature_labels(self):
        return ['min:' + s for s in self.elements]


class Compositions(BaseDescriptor):
    """
    Calculate elemental descriptors from compound's composition.
    """

    classic = ['WeightedAverage', 'WeightedSum', 'WeightedVariance', 'MaxPooling', 'MinPooling']

    def __init__(self,
                 *,
                 elemental_info: Union[pd.DataFrame, None] = None,
                 n_jobs: int = -1,
                 featurizers: Union[str, List[str]] = 'classic',
                 on_errors: str = 'nan'):
        """

        Parameters
        ----------
        elemental_info
            Elemental level information for each element. For example, the ``atomic number``,
            ``atomic radius``, and etc. By default (``None``), will use the XenonPy embedded information.
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
        featurizers: Union[str, List[str]]
            Name of featurizers that will be used.
            Set to `classic` to be compatible with the old version.
            This is equal to set ``featurizers=['WeightedAverage', 'WeightedSum',
            'WeightedVariance', 'MaxPooling', 'MinPooling']``.
            Default is 'all'.
        on_errors: string
            How to handle exceptions in feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'nan' which will raise up the exception.
        """

        if featurizers == 'classic':
            super().__init__(featurizers=self.classic)
        else:
            super().__init__(featurizers=featurizers)

        self.composition = Counting(n_jobs=n_jobs, on_errors=on_errors)
        self.composition = WeightedAverage(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = WeightedSum(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = WeightedVariance(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = GeometricMean(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = HarmonicMean(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = MaxPooling(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
        self.composition = MinPooling(n_jobs=n_jobs, on_errors=on_errors, elemental_info=elemental_info)
