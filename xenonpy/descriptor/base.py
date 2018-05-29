# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


import sys
import traceback
import types
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

from ..utils.functional import TimedMetaClass


class BaseFeaturizer(BaseEstimator, TransformerMixin):
    """
    Abstract class to calculate features from raw materials input data
    such a compound formula or a pymatgen crystal structure or
    bandstructure object.
    """

    def __init__(self, n_jobs=-1,
                 ignore_errors=False,
                 return_errors=False):
        """
        Parameters
        ----------
        n_jobs: int
            The number of jobs to run in parallel for both fit and predict.
            If -1, then the number of jobs is set to the number of cores.
        ignore_errors: bool
            Returns NaN for entries where exceptions are
            thrown if True. If False, exceptions are thrown as normal.
        return_errors: bool
            If True, returns the feature list as
            determined by ignore_errors with traceback strings added
            as an extra 'feature'. Entries which featurize without
            exceptions have this extra feature set to NaN.
        """
        self._n_jobs = cpu_count() if n_jobs is -1 else n_jobs
        if return_errors and not ignore_errors:
            raise ValueError("Please set ignore_errors to True to use"
                             " return_errors.")
        self.__ignore_errors = ignore_errors
        self.__return_errors = return_errors

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, n_jobs):
        """Set the number of threads for this """
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
        if not isinstance(entries, pd.Series):
            raise Exception("'entries' must be a <pd.Series> object")

        # Special case: Empty list
        if len(entries) is 0:
            return []

        # keep index
        index = entries.index

        # If the featurize function only has a single arg, zip the inputs
        if not isinstance(entries[0], (tuple, list, np.ndarray)):
            entries = zip(entries)

        # Run the actual featurization
        if self.n_jobs == 1:
            ret = [self._wrapper(x) for x in entries]
        else:
            with Pool(self.n_jobs) as p:
                ret = p.map(self._wrapper, entries)

        labels = None
        try:
            labels = self.feature_labels()
        except NotImplementedError:
            pass

        return pd.DataFrame(ret, index=index, columns=labels)

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
            if self.__return_errors:
                return self.featurize(*x), None
            else:
                return self.featurize(*x)
        except BaseException as e:
            if self.__ignore_errors:
                if self.__return_errors:
                    features = [float("nan")] * len(self.feature_labels())
                    error = traceback.format_exception(*sys.exc_info())
                    return features, "".join(error)
                else:
                    return [float("nan")] * len(self.feature_labels())
            else:
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

    def feature_labels(self):
        """
        Generate attribute names.
        Returns:
            ([str]) attribute labels.
        """

        raise NotImplementedError("feature_labels() is not defined!")

    def citations(self):
        """
        Citation(s) and reference(s) for this feature.
        Returns:
            (list) each element should be a string citation,
                ideally in BibTeX format.
        """

        raise NotImplementedError("citations() is not defined!")

    def implementors(self):
        """
        List of implementors of the feature.
        Returns:
            (list) each element should either be a string with author name (e.g.,
                "Anubhav Jain") or a dictionary  with required key "name" and other
                keys like "email" or "institution" (e.g., {"name": "Anubhav
                Jain", "email": "ajain@lbl.gov", "institution": "LBNL"}).
        """

        raise NotImplementedError("implementors() is not defined!")


class BaseDescriptor(BaseEstimator, TransformerMixin, metaclass=TimedMetaClass):
    n_jobs = cpu_count()

    @property
    def elapsed(self):
        return self._timer.elapsed

    def __setattr__(self, key, value):

        if '__features__' not in self.__dict__:
            super().__setattr__('__features__', defaultdict(list))
        try:
            if isinstance(value, TransformerMixin) and isinstance(value.featurize, types.MethodType):
                self.__features__[key].append(value)
        except AttributeError:
            pass
        super().__setattr__(key, value)

    def _if_series(self, o):
        if isinstance(o, pd.Series):
            if len(self.__features__) > 1 or not o.name or o.name not in self.__features__:
                raise KeyError('Pandas Series object must have name corresponding to feature type name')
            return True
        if isinstance(o, pd.DataFrame):
            for k in self.__features__:
                if k not in o:
                    raise KeyError('Pandas Series object must have name corresponding to feature type name')
            return False
        raise TypeError('X, y must be <pd.DataFrame> or <pd.Series>')

    def fit(self, X, y=None, **fit_params):

        if y and not isinstance(y, pd.Series):
            raise TypeError('y must be <pd.Series> or None')

        if self._if_series(X):
            features = self.__features__[X.name]
            for f in features:
                f.fit(X, y, **fit_params)
        else:
            for col in X:
                features = self.__features__[col]
                for f in features:
                    f.fit(X[col], y, **fit_params)

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

        if self._if_series(X):
            features = self.__features__[X.name]
            for f in features:
                ret = f.transform(X)
                if isinstance(ret, list):
                    ret = _make_df(f, ret)
                results.append(ret)

        else:
            for col in X:
                features = self.__features__[col]
                for f in features:
                    ret = f.transform(X[col])
                    if isinstance(ret, list):
                        ret = _make_df(f, ret)
                    results.append(ret)

        return pd.concat(results, axis=1)
