# Copyright 2017 TsumiNa. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class BaseFeature(BaseEstimator, TransformerMixin):
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

    def transform(self, X):
        """
        Featurize a list of entries.
        If `featurize` takes multiple inputs, supply inputs as a list of tuples.
        Args
        ----
        X: list-like
            A list of entries to be featurized.

        Returns
        -------
            DataFrame
                features for each entry.
        """

        # Check inputs
        if not hasattr(X, '__getitem__'):
            raise Exception("'entries' must be a list-like object")

        # Special case: Empty list
        if len(X) is 0:
            return []

        # If the featurize function only has a single arg, zip the inputs
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        # Run the actual featurization
        if self.n_jobs == 1:
            ret = [self._wrapper(x) for x in X]
        else:
            with Pool(self.n_jobs) as p:
                ret = p.map(self._wrapper, X)

        labels = self.feature_labels()
        if isinstance(X, pd.Series):
            index = X.index
        else:
            index = [str(i) for i in range(len(X))]
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
            return self.featurize(x)
        except BaseException as e:
            if self.__ignore_errors:
                return np.nan
            raise e

    def featurize(self, x):
        """
        Main featurizer function, which has to be implemented
        in any derived featurizer subclass.
        Args:
            x: input data to featurize (type depends on featurizer).
        Returns:
            (list) one or more features.
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

