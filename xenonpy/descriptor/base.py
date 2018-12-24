# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from collections import defaultdict
from collections.abc import Iterable
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from ..utils import TimedMetaClass


class BaseFeaturizer(BaseEstimator, TransformerMixin):
    """
    Abstract class to calculate features from :class:`pandas.Series` input data.
    Each entry can be any format such a compound formula or a pymatgen crystal structure
    dependent on the featurizer implementation.

    This class have similar structure with `matminer BaseFeaturizer`_ but follow more strict convention.
    That means you can embed this feature directly into `matminer BaseFeaturizer`_ class implement.::

        class MatFeature(BaseFeaturizer):
            def featurize(self, *x):
                return <xenonpy_featurizer>.featurize(*x)

    .. _matminer BaseFeaturizer: https://github.com/hackingmaterials/matminer/blob/master/matminer/featurizers/base.py

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
    _n_jobs = 1

    def __init__(self, n_jobs=-1, ignore_errors=False):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            If -1, then the number of jobs is set to the number of cores.
        ignore_errors: bool
            Returns NaN for entries where exceptions are
            thrown if True. If False, exceptions are thrown as normal.
        """
        self.n_jobs = n_jobs
        self.__ignore_errors = ignore_errors

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        """Set the number of threads for this """
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

    def transform(self, entries):
        """
        Featurize a list of entries.
        If `featurize` takes multiple inputs, supply inputs as a list of tuples.
        Args
        ----
        entries: list-like
            A list of entries to be featurized.

        Returns
        -------
            DataFrame
                features for each entry.
        """

        # Check inputs
        if not isinstance(entries, Iterable):
            raise TypeError('parameter "entries" must be a iterable object')

        # Special case: Empty list
        if len(entries) is 0:
            return []

        # Run the actual featurization
        if self.n_jobs == 1:
            ret = [self._wrapper(x) for x in entries]
        else:
            with Pool(self.n_jobs) as p:
                ret = p.map(self._wrapper, entries)

        try:
            labels = self.feature_labels
        except NotImplementedError:
            labels = None

        if isinstance(entries, pd.Series):
            return pd.DataFrame(ret, index=entries.index, columns=labels)
        if isinstance(entries, np.ndarray):
            return np.array(ret)
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
                return self.featurize(x)
            return self.featurize(*x)
        except BaseException as e:
            if self.__ignore_errors:
                return [float("nan")] * len(self.feature_labels)
            raise e

    def featurize(self, *x):
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

        raise NotImplementedError("featurize() is not defined!")

    @property
    def feature_labels(self):
        """
        Generate attribute names.
        Returns:
            ([str]) attribute labels.
        """
        raise NotImplementedError("feature_labels() is not defined!")

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

    _n_jobs = 1

    def __init__(self):
        self.__featurizers__ = defaultdict(list)

    @property
    def elapsed(self):
        return self._timer.elapsed

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        """Set the number of threads for this """
        if n_jobs > cpu_count() or n_jobs == -1:
            self._n_jobs = cpu_count()
        else:
            self._n_jobs = n_jobs

    def __setattr__(self, key, value):

        if key == '__featurizers__':
            if not isinstance(value, defaultdict):
                raise RuntimeError('Can not set "__featurizers__" yourself')
            super().__setattr__(key, value)
        if isinstance(value, BaseFeaturizer):
            self.__featurizers__[key].append(value)
        else:
            super().__setattr__(key, value)

    def __repr__(self):
        return super().__repr__() + ':\n' + \
               '\n'.join(['  |- %s:\n  |  |- %s' % (k, '\n  |  |- '.join(map(str, v))) for k, v in
                          self.__featurizers__.items()])

    def _check_input(self, o):
        if isinstance(o, (list, np.ndarray)):
            keys = list(self.__featurizers__.keys())
            if len(keys) == 1:
                return pd.DataFrame(o, columns=[keys[0]])
            raise TypeError(
                'column name of Seriers/DataFrame must corresponding to featurizer name')

        if isinstance(o, pd.Series):
            if o.name in self.__featurizers__:
                return pd.DataFrame(o)
            raise KeyError('Pandas Series object must have name corresponding to feature type name')

        if isinstance(o, pd.DataFrame):
            for k in o:
                if k not in self.__featurizers__:
                    raise KeyError(
                        'Pandas Series object must have name corresponding to feature <%s>' % k)
            return o
        raise TypeError('X, y must be <list>, <numpy.array>, <pd.DataFrame> or <pd.Series>')

    def _map_name(self, **fit_params):
        for k in fit_params:
            if k in self.__featurizers__:
                print(k)
                self.__featurizers__[fit_params[k]] = self.__featurizers__.pop(k)

    @property
    def featurizers(self):
        return self.__featurizers__.keys()

    def fit(self, X, y=None, **fit_params):
        self._map_name(**fit_params)

        X = self._check_input(X)
        for k, features in self.__featurizers__.items():
            for f in features:
                f.fit(X[k], y, **fit_params)

        return self

    def transform(self, X):

        def _make_df(feature_, ret_):
            try:
                labels = feature_.feature_labels()
            except NotImplementedError:
                labels = None
            return pd.DataFrame(ret_, index=X.index, columns=labels)

        if len(X) is 0:
            return None

        results = []

        X = self._check_input(X)
        for k, features in self.__featurizers__.items():
            for f in features:
                ret = f.transform(X[k])
                if isinstance(ret, list):
                    ret = _make_df(f, ret)
                results.append(ret)

        return pd.concat(results, axis=1)
