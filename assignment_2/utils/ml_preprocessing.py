"""Reusable preprocessing pieces for the model pipeline.

Everything here LEARNS from data in fit() (percentile limits), so it must only ever be fitted on training rows.
It lives in its own module so that a pickled model can be loaded later (inference, Airflow) with the same code.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class PercentileClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [lower, upper] percentiles learned on the training data.

    Tames long tails (e.g. balance_to_debt) without deleting rows. NaNs are left alone for the imputer.
    """

    def __init__(self, lower=0.5, upper=99.5):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanpercentile(X, self.lower, axis=0)
        self.high_ = np.nanpercentile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
