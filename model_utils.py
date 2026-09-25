import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

MIN_MONTHS = 1.0


class ClippedLinearRegression(BaseEstimator, RegressorMixin):
    """LinearRegression whose predictions are floored at MIN_MONTHS.
    Survival time can't be <= 0, so this replaces the notebook's
    post-hoc max(pred, 1) with a proper estimator-level fix."""

    def __init__(self, floor=MIN_MONTHS):
        self.floor = floor

    def fit(self, X, y):
        self.model_ = LinearRegression().fit(X, y)
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        return self

    def predict(self, X):
        return np.clip(self.model_.predict(X), self.floor, None)