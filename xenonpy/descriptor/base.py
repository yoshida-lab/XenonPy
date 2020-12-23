#  Copyright (c) 2019. yoshida-lab. All rights reserved.
#  Use of this source code is governed by a BSD-style
#  license that can be found in the LICENSE file.

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from copy import copy
import itertools
import warnings
from multiprocessing import cpu_count
from typing import DefaultDict, List, Sequence, Union, Set
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
from pymatgen.core.composition import Composition as PMGComp
from sklearn.base import TransformerMixin, BaseEstimator

from xenonpy.datatools.preset import preset
from xenonpy.utils import TimedMetaClass, Switch


class BaseFeaturizer(BaseEstimator, TransformerMixin, metaclass=ABCMeta):
    """
    Abstract class to calculate features from :class:`pandas.Series` input data.
    Each entry can be any format such a compound formula or a pymatgen crystal structure
    dependent on the featurizer implementation.

    This class have similar structure with `matminer BaseFeaturizer`_ but follow more strict convention.
    That means you can embed this feature directly into `matminer BaseFeaturizer`_ class implement.::

        class MatFeature(BaseFeaturizer):
            def featurize(self, *x):
                return <xenonpy_featurizer>.featurize(*x)

    .. _matminer BaseFeaturizer: https://github.com/hackingmaterials/matminer/blob/master/matminer/featurizers/base_smc.py

    **Using a BaseFeaturizer Class**

    :meth:`BaseFeaturizer` implement :class:`sklearn.base.BaseEstimator` and :class:`sklearn.base.TransformerMixin`
    that means you can use it in a scikit-learn way.::

        featurizer = SomeFeaturizer()
        features = featurizer.fit_transform(X)

    You can also employ the featurizer as part of a ScikitLearn Pipeline object.
    You would then provide your input data as an array to the Pipeline, which would
    output the featurers as an :class:`pandas.DataFrame`.

    :class:`BaseFeaturizer` also provide you to retrieving proper references for a featurizer.
    The ``__citations__`` returns a list of papers that should be cited.
    The ``__authors__`` returns a list of people who wrote the featurizer.
    Also can be accessed from property ``citations`` and ``citations``.

    **Implementing a New BaseFeaturizer Class**

    These operations must be implemented for each new featurizer:

    - ``featurize`` - Takes a single material as input, returns the features of that material.
    - ``feature_labels`` - Generates a human-meaningful name for each of the features. **Implement this as property**.

    Also suggest to implement these two **properties**:

    - ``citations`` - Returns a list of citations in BibTeX format.
    - ``implementors`` - Returns a list of people who contributed writing a paper.

    All options of the featurizer must be set by the ``__init__`` function. All
    options must be listed as keyword arguments with default values, and the
    value must be saved as a class attribute with the same name or as a property
    (e.g., argument `n` should be stored in `self.n`).
    These requirements are necessary for
    compatibility with the ``get_params`` and ``set_params`` methods of ``BaseEstimator``,
    which enable easy interoperability with scikit-learn.
    :meth:`featurize` must return a list of features in :class:`numpy.ndarray`.

    .. note::

        None of these operations should change the state of the featurizer. I.e.,
        running each method twice should no produce different results, no class
        attributes should be changed, unning one operation should not affect the
        output of another.

    """

    __authors__ = ['anonymous']
    __citations__ = ['No citations']

    def __init__(
        self,
        n_jobs: int = -1,
        *,
        on_errors: str = 'raise',
        return_type: str = 'any',
        target_col: Union[List[str], str, None] = None,
        parallel_verbose: int = 0,
    ):
        """
        Parameters
        ----------
        n_jobs
            The number of jobs to run in parallel for both fit and predict. Set -1 to use all cpu cores (default).
            Inputs ``X`` will be split into some blocks then run on each cpu cores.
            When set to 0, input X will be treated as a block and pass to ``Featurizer.featurize`` directly.
            This default parallel implementation does not support pd.DataFrame input,
            so please make sure you set n_jobs=0 if the input will be pd.DataFrame.
        on_errors
            How to handle the exceptions in a feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        return_type
            Specify the return type.
            Can be ``any``, ``custom``, ``array`` and ``df``.
            ``array`` and ``df`` force return type to ``np.ndarray`` and ``pd.DataFrame`` respectively.
            If ``any`` or ``custom``, the return type depends on multiple factors (see transform function).
            Default is ``any``
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            If ``None``, all columns of the pd.DataFrame is used.
            Default is None.
        parallel_verbose
            The verbosity level: if non zero, progress messages are printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported. Default ``0``.
        """
        self.return_type = return_type
        self.target_col = target_col
        self.n_jobs = n_jobs
        self.on_errors = on_errors
        self.parallel_verbose = parallel_verbose
        self._kwargs = {}

    @property
    def return_type(self):
        return self._return_type

    @return_type.setter
    def return_type(self, val):
        if val not in {'any', 'array', 'df', 'custom'}:
            raise ValueError('`return_type` must be `any`, `custom`, `array` or `df`')
        self._return_type = val

    @property
    def on_errors(self):
        return self._on_errors

    @on_errors.setter
    def on_errors(self, val):
        if val not in {'nan', 'keep', 'raise'}:
            raise ValueError('`on_errors` must be `nan`, `keep` or `raise`')
        self._on_errors = val

    @property
    def parallel_verbose(self):
        return self._parallel_verbose

    @parallel_verbose.setter
    def parallel_verbose(self, val):
        if not isinstance(val, int):
            raise ValueError('`parallel_verbose` must be int')
        self._parallel_verbose = val

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        """Set the number of threads for this """
        if n_jobs < -1:
            n_jobs = -1
        if n_jobs > cpu_count() or n_jobs == -1:
            self._n_jobs = cpu_count()
        else:
            self._n_jobs = n_jobs

    def fit(self, X, y=None, **fit_kwargs):
        """Update the parameters of this featurizer based on available data
        Args:
            X - [list of tuples], training data
        Returns:
            self
            """
        return self

    # todo: Dose fit_transform need to pass paras to transform?
    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X, **fit_params)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X, **fit_params)

    def transform(self, entries: Sequence, *, return_type=None, target_col=None, **kwargs):
        """
        Featurize a list of entries.
        If `featurize` takes multiple inputs, supply inputs as a list of tuples,
        or use pd.DataFrame with parameter ``target_col`` to specify the column name(s).
        
        Args
        ----
        entries: list-like or pd.DataFrame
            A list of entries to be featurized or pd.DataFrame with one specified column.
            See detail of target_col if entries is pd.DataFrame.
            Also, make sure n_jobs=0 for pd.DataFrame.
        return_type: str
            Specify the return type.
            Can be ``any``, ``custom``, ``array`` or ``df``.
            ``array`` or ``df`` forces return type to ``np.ndarray`` or ``pd.DataFrame``, respectively.
            If ``any``, the return type follow prefixed rules:
            (1) if input type is pd.Series or pd.DataFrame, returns pd.DataFrame;
            (2) else if input type is np.array, returns np.array;
            (3) else if other input type and n_jobs=0, follows the featurize function return;
            (4) otherwise, return a list of objects (output of featurize function).
            If ``custom``, the return type depends on the featurize function if n_jobs=0,
            or the return type is a list of objects (output of featurize function) for other n_jobs values.
            This is a one-time change that only have effect in the current transformation.
            Default is ``None`` for using the setting at initialization step.
        target_col
            Only relevant when input is pd.DataFrame, otherwise ignored.
            Specify a single column to be used for transformation.
            Default is ``None`` for using the setting at initialization step.
            (see __init__ for more information)

        Returns
        -------
            DataFrame
                features for each entry.
        """
        self._kwargs = kwargs

        # Check inputs
        if not isinstance(entries, Iterable):
            raise TypeError('parameter "entries" must be a iterable object')

        # Extract relevant columns for pd.DataFrame input
        if isinstance(entries, pd.DataFrame):
            if target_col is None:
                target_col = self.target_col
                if target_col is None:
                    target_col = entries.columns.values
            entries = entries[target_col]

        # Special case: Empty list
        if len(entries) is 0:
            return []

        # Check outputs
        if return_type not in {None, 'any', 'array', 'df', 'custom'}:
            raise ValueError('`return_type` must be None, `any`, `custom`, `array` or `df`')

        for c in Switch(self._n_jobs):
            if c(0):
                # Run the actual featurization
                ret = self.featurize(entries, **kwargs)
                break
            if isinstance(entries, pd.DataFrame):
                raise RuntimeError(
                    "Auto-parallel can not be used when`entries` is `pandas.DataFrame`. "
                    "Please set `n_jobs` to 0 and implements your algorithm in the `featurize` method"
                )
            if c(1):
                ret = [self._wrapper(x) for x in entries]
                break
            if c():
                ret = Parallel(n_jobs=self._n_jobs, verbose=self._parallel_verbose)(
                    delayed(self._wrapper)(x) for x in entries)

        try:
            labels = self.feature_labels
        except NotImplementedError:
            labels = None

        if return_type is None:
            return_type = self.return_type

        if return_type == 'any':
            if isinstance(entries, (pd.Series, pd.DataFrame)):
                tmp = pd.DataFrame(ret, index=entries.index, columns=labels)
                return tmp
            if isinstance(entries, np.ndarray):
                return np.array(ret)
            return ret

        if return_type == 'array':
            return np.array(ret)

        if return_type == 'df':
            if isinstance(entries, (pd.Series, pd.DataFrame)):
                return pd.DataFrame(ret, index=entries.index, columns=labels)
            return pd.DataFrame(ret, columns=labels)

        if return_type == 'custom':
            return ret

    def _wrapper(self, x):
        """
        An exception wrapper for featurize, used in featurize_many and
        featurize_dataframe. featurize_wrapper changes the behavior of featurize
        when ignore_errors is True in featurize_many/dataframe.
        Args:
             x: input data to featurize (type depends on featurizer).
        Returns:
            (list) one or more features.
        """
        try:
            # Successful featurization returns nan for an error.
            if not isinstance(x, (tuple, list, np.ndarray)):
                return self.featurize(x, **self._kwargs)
            return self.featurize(*x, **self._kwargs)
        except Exception as e:
            if self._on_errors == 'nan':
                return [np.nan] * len(self.feature_labels)
            elif self._on_errors == 'keep':
                return [e] * len(self.feature_labels)
            else:
                raise e

    @abstractmethod
    def featurize(self, *x, **kwargs):
        """
        Main featurizer function, which has to be implemented
        in any derived featurizer subclass.

        Args
        ====
        x: depends on featurizer
            input data to featurize.

        Returns
        =======
        any: numpy.ndarray
            one or more features.
        """

    @property
    @abstractmethod
    def feature_labels(self):
        """
        Generate attribute names.
        Returns:
            ([str]) attribute labels.
        """

    @property
    def citations(self):
        """
        Citation(s) and reference(s) for this feature.
        Returns:
            (list) each element should be a string citation,
                ideally in BibTeX format.
        """
        return '\n'.join(self.__citations__)

    @property
    def authors(self):
        """
        List of implementors of the feature.
        Returns:
            (list) each element should either be a string with author name (e.g.,
                "Anubhav Jain") or a dictionary  with required key "name" and other
                keys like "email" or "institution" (e.g., {"name": "Anubhav
                Jain", "email": "ajain@lbl.gov", "institution": "LBNL"}).
        """

        return '\n'.join(self.__authors__)


class BaseDescriptor(BaseEstimator, TransformerMixin, metaclass=TimedMetaClass):
    """
    Abstract class to organize featurizers.
    This class can take list-like[object] or pd.DataFrame as input for transformation or fitting.
    For pd.DataFrame, if any column name matches any group name,
    the matched group(s) will be calculated with corresponding column(s);
    otherwise, the pd.DataFrame will be passed on as is.

    Examples
    --------

    .. code::


        class MyDescriptor(BaseDescriptor):

            def __init__(self, n_jobs=-1):
                self.descriptor = SomeFeature1(n_jobs)
                self.descriptor = SomeFeature2(n_jobs)
                self.descriptor = SomeFeature3(n_jobs)
                self.descriptor = SomeFeature4(n_jobs)

    """

    def __init__(self, *, featurizers: Union[List[str], str] = 'all', on_errors: str = 'raise'):
        """

        Parameters
        ----------
        featurizers
            Specify which Featurizer(s) will be used.
            Default is 'all'.
        on_errors
            How to handle the exceptions in a feature calculations. Can be 'nan', 'keep', 'raise'.
            When 'nan', return a column with ``np.nan``.
            The length of column corresponding to the number of feature labs.
            When 'keep', return a column with exception objects.
            The default is 'raise' which will raise up the exception.
        """
        self.__featurizers__: Set[str] = set()  # featurizers' names
        self.__featurizer_sets__: DefaultDict[str, List[BaseFeaturizer]] = defaultdict(list)
        self.featurizers = featurizers
        self.on_errors = on_errors

    @property
    def on_errors(self):
        return self._on_errors

    @on_errors.setter
    def on_errors(self, val):
        if val not in {'nan', 'keep', 'raise'}:
            raise ValueError('`on_errors` must be `nan`, `keep` or `raise`')
        self._on_errors = val
        for fea_set in self.__featurizer_sets__.values():
            for fea in fea_set:
                fea.on_errors = val

    @property
    def featurizers(self):
        return self._featurizers

    @featurizers.setter
    def featurizers(self, val):
        if isinstance(val, str):
            if val != 'all':
                self._featurizers = (val,)
            else:
                self._featurizers = val
        elif isinstance(val, (tuple, List)):
            self._featurizers = tuple(val)
        else:
            raise ValueError(
                'parameter `featurizers` must be `all`, name of featurizer, or list of name of featurizer'
            )

    @property
    def elapsed(self):
        return self._timer.elapsed

    def __setattr__(self, key, value):

        if key == '__featurizer_sets__':
            if not isinstance(value, defaultdict):
                raise RuntimeError('Can not set "self.__featurizer_sets__" by yourself')
            super().__setattr__(key, value)
        if isinstance(value, BaseFeaturizer):
            if value.__class__.__name__ in self.__featurizers__:
                raise RuntimeError('Duplicated featurizer <%s>' % value.__class__.__name__)
            self.__featurizer_sets__[key].append(value)
            self.__featurizers__.add(value.__class__.__name__)
        else:
            super().__setattr__(key, value)

    def __repr__(self):
        return self.__class__.__name__ + ':\n' + \
               '\n'.join(
                   ['  |- %s:\n  |  |- %s' % (k, '\n  |  |- '.join(map(lambda s: s.__class__.__name__, v))) for k, v in
                    self.__featurizer_sets__.items()])

    def _check_input(self, X, y=None, **kwargs):

        def _reformat(x):
            if x is None:
                return x

            keys = list(self.__featurizer_sets__.keys())
            if len(keys) == 1:
                if isinstance(x, list):
                    return pd.DataFrame(pd.Series(x), columns=keys)

                if isinstance(x, np.ndarray):
                    if len(x.shape) == 1:
                        return pd.DataFrame(x, columns=keys)

                if isinstance(x, pd.Series):
                    return pd.DataFrame(x.values, columns=keys, index=x.index)

            if isinstance(x, pd.Series):
                x = pd.DataFrame(x)

            if isinstance(x, pd.DataFrame):
                tmp = set(x.columns) | set(kwargs.keys())
                if set(keys).isdisjoint(tmp):
                    # raise KeyError('name of columns do not match any feature set')
                    warnings.warn(
                        'name of columns do not match any feature set, '
                        'the whole dataframe is applied to all feature sets', UserWarning)
                    # allow type check later for this special case
                    return [x]
                return x

            raise TypeError('you can not ues a array-like input '
                            'because there are multiple feature sets or the dim of input is not 1')

        return _reformat(X), _reformat(y)

    def _rename(self, **fit_params):
        for k, v in fit_params.items():
            if k in self.__featurizer_sets__:
                self.__featurizer_sets__[v] = self.__featurizer_sets__.pop(k)

    @property
    def all_featurizers(self):
        return list(self.__featurizers__)

    def fit(self, X, y=None, **kwargs):
        if not isinstance(X, Iterable):
            raise TypeError('parameter "entries" must be a iterable object')

        self._rename(**kwargs)

        # assume y is in same format of X (do not cover other cases now)
        X, y = self._check_input(X, y)
        if isinstance(X, list):
            for k, features in self.__featurizer_sets__.items():
                for f in features:
                    if self._featurizers != 'all' and f.__class__.__name__ not in self._featurizers:
                        continue
                    # assume y is in same format of X
                    if y is not None:
                        f.fit(X[0], y[0], **kwargs)
                    else:
                        f.fit(X[0], **kwargs)
        else:
            for k, features in self.__featurizer_sets__.items():
                if k in X:
                    for f in features:
                        if self._featurizers != 'all' and f.__class__.__name__ not in self._featurizers:
                            continue
                        if y is not None and k in y:
                            f.fit(X[k], y[k], **kwargs)
                        else:
                            f.fit(X[k], **kwargs)

        return self

    def transform(self, X, **kwargs):
        if not isinstance(X, Iterable):
            raise TypeError('parameter "entries" must be a iterable object')

        if len(X) is 0:
            return None

        if 'return_type' in kwargs:
            del kwargs['return_type']

        results = []

        X, _ = self._check_input(X, **kwargs)
        if isinstance(X, list):
            for k, features in self.__featurizer_sets__.items():
                # if k in kwargs:
                #     k = kwargs[k]
                for f in features:
                    if self._featurizers != 'all' and f.__class__.__name__ not in self._featurizers:
                        continue
                    ret = f.transform(X[0], return_type='df', **kwargs)
                    results.append(ret)
        else:
            for k, features in self.__featurizer_sets__.items():
                if k in kwargs:
                    k = kwargs[k]
                if k in X:
                    for f in features:
                        if self._featurizers != 'all' and f.__class__.__name__ not in self._featurizers:
                            continue
                        ret = f.transform(X[k], return_type='df', **kwargs)
                        results.append(ret)

        return pd.concat(results, axis=1)

    @property
    def feature_labels(self):
        """
        Generate attribute names.
        Returns:
            ([str]) attribute labels.
        """

        if len(self.__featurizers__) == 0:
            raise NotImplementedError("no featurizers")

        ret = ()
        for k, features in self.__featurizer_sets__.items():
            ret += ((k, list(itertools.chain.from_iterable([f.feature_labels for f in features]))),)

        if len(ret) == 1:
            return ret[0][1]
        return ret


class BaseCompositionFeaturizer(BaseFeaturizer, metaclass=ABCMeta):

    def __init__(self,
                 *,
                 elemental_info: Union[pd.DataFrame, None] = None,
                 n_jobs: int = -1,
                 on_errors: str = 'raise',
                 return_type: str = 'any',
                 target_col: Union[List[str], str, None] = None):
        """
        Base class for composition feature.
        """

        super().__init__(n_jobs=n_jobs,
                         on_errors=on_errors,
                         return_type=return_type,
                         target_col=target_col)

        if elemental_info is None:
            self._elements = preset.elements_completed
        else:
            self._elements = elemental_info
        self.__authors__ = ['TsumiNa']

    def featurize(self, comp):
        elems_, nums_ = [], []
        if isinstance(comp, PMGComp):
            comp = comp.as_dict()
        for e, n in comp.items():
            elems_.append(e)
            nums_.append(n)
        return self.mix_function(elems_, nums_)

    @abstractmethod
    def mix_function(self, elems, nums):
        """

        Parameters
        ----------
        elems: list
            Elements in compound.
        nums: list
            Number of each element.

        Returns
        -------
        descriptor: numpy.ndarray
        """
