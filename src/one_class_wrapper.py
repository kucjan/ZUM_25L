import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.validation import check_is_fitted

from config.constants import (
    EXTEND_FACTOR,
    EXTREME_FACTOR,
    EXTREME_SCALER,
    MIN_UNIQUE_NUM_VALUES,
    OUTLIER_RATIO,
)
from config.types import ArrayLike, MatrixLike


class OneClassWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: BaseEstimator,
        outlier_ratio: float = OUTLIER_RATIO,
        extend_factor: float = EXTEND_FACTOR,
        extreme_factor: float = EXTREME_FACTOR,
        extreme_scaler: int = EXTREME_SCALER,
    ) -> None:
        """Initializes the OneClassWrapper.

        Args:
            model (BaseEstimator): A scikit-learn compatible binary classification model.
            outlier_ratio (float, optional): Ratio of outliers to be added. Defaults to OUTLIER_RATIO.
            extend_factor (float, optional): Factor by which to extend the outlier value range. Defaults to EXTEND_FACTOR.
            extreme_factor (float, optional): Determines chances for generating extreme values. Defaults to EXTREME_FACTOR.
            extreme_scaler (int, optional): Scaling multiplier for generating outlier ranges. Defaults to EXTREME_SCALER.
        """
        self.model = model
        self.outlier_ratio = outlier_ratio
        self.extend_factor = extend_factor
        self.extreme_factor = extreme_factor
        self.extreme_scaler = extreme_scaler

    def _determine_column_types(self, X: pd.DataFrame) -> dict:
        """Determines whether each column is numerical or categorical.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            dict: Dictionary mapping column names to their determined types ('numerical' or 'categorical').
        """
        col_types = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                if X[col].nunique() <= MIN_UNIQUE_NUM_VALUES:
                    col_types[col] = "categorical"
                else:
                    col_types[col] = "numerical"
            else:
                col_types[col] = "categorical"
        return col_types

    def _generate_numerical_outliers(
        self, col: pd.Series, n_samples: int
    ) -> np.ndarray:
        """Generates numerical outlier values for a given column.

        Args:
            col (pd.Series): Numerical column from the dataset.
            n_samples (int): Number of outlier samples to generate.

        Returns:
            np.ndarray: Array of generated outlier values.
        """
        min_val, max_val = col.min(), col.max()
        range_val = max_val - min_val
        scale = range_val * self.extreme_scaler

        outliers = np.empty(n_samples)

        lower_outliers = np.random.uniform(
            low=min_val - scale,
            high=min_val + range_val * self.extend_factor,
            size=n_samples // 2,
        )
        upper_outliers = np.random.uniform(
            low=max_val - range_val * self.extend_factor,
            high=max_val + scale,
            size=n_samples - (n_samples // 2),
        )

        outliers[: len(lower_outliers)] = lower_outliers
        outliers[len(lower_outliers) :] = upper_outliers

        np.random.shuffle(outliers)
        return outliers

    def _generate_categorical_outliers(
        self, col: pd.Series, n_samples: int
    ) -> np.ndarray:
        """Generates categorical outlier values by sampling existing unique values.

        Args:
            col (pd.Series): Categorical column from the dataset.
            n_samples (int): Number of outlier samples to generate.

        Returns:
            np.ndarray: Array of sampled outlier values.
        """
        unique_vals = col.unique()
        all_options = unique_vals
        return np.random.choice(all_options, size=n_samples)

    def _generate_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generates an outlier dataset based on the input features.

        Args:
            X (pd.DataFrame): Original dataset.

        Returns:
            pd.DataFrame: Dataset of synthetic outlier samples.
        """
        n_outliers = int(len(X) * self.outlier_ratio)
        col_types = self._determine_column_types(X)
        outliers = pd.DataFrame()
        for col in X.columns:
            if col_types[col] == "numerical":
                outliers[col] = self._generate_numerical_outliers(X[col], n_outliers)
            else:
                outliers[col] = self._generate_categorical_outliers(X[col], n_outliers)
        return outliers

    def _prepare_data(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Combines original data with synthetic outliers and encodes categorical features.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of combined feature matrix and labels.
        """
        outliers = self._generate_outliers(X)
        y_combined = np.concatenate([np.ones(len(X)), np.zeros(len(outliers))])
        X_combined = pd.concat([X, outliers], axis=0).reset_index(drop=True)
        self._encoder = {}
        for col in X_combined.columns:
            if not pd.api.types.is_numeric_dtype(X_combined[col]):
                le = LabelEncoder()
                X_combined[col] = le.fit_transform(X_combined[col])
                self._encoder[col] = le
        return X_combined.values, y_combined

    def fit(self, X: pd.DataFrame) -> "OneClassWrapper":
        """Fits the wrapped model on a dataset augmented with synthetic outliers.

        Args:
            X (pd.DataFrame): Input data to train the model.

        Returns:
            OneClassWrapper: Fitted instance of this class.
        """
        self.X_combined, self.y_combined = self._prepare_data(X)
        self.model.fit(self.X_combined, self.y_combined)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class labels for the given input data.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            np.ndarray: Predicted class labels.
        """
        check_is_fitted(self, "is_fitted_")
        X_encoded = X.copy()
        if hasattr(self, "_encoder"):
            for col in X_encoded.columns:
                if col in self._encoder:
                    X_encoded[col] = self._encoder[col].transform(X_encoded[col])
        return self.model.predict(X_encoded.values)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts class probabilities for the given input data.

        Args:
            X (pd.DataFrame): Input data.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        check_is_fitted(self, "is_fitted_")
        X_encoded = X.copy()
        if hasattr(self, "_encoder"):
            for col in X_encoded.columns:
                if col in self._encoder:
                    X_encoded[col] = self._encoder[col].transform(X_encoded[col])
        return self.model.predict_proba(X_encoded.values)

    def score(self, X: MatrixLike, y: MatrixLike | ArrayLike) -> float:
        """Calculates the accuracy score of the model.

        Args:
            X (MatrixLike): Input features.
            y (MatrixLike | ArrayLike): True labels.

        Returns:
            float: Accuracy score.
        """
        return self.evaluate(X, y)["accuracy"]

    def evaluate(self, X: MatrixLike, y: MatrixLike | ArrayLike) -> dict:
        """Evaluates the model using accuracy, precision, recall, and F1-score.

        Args:
            X (MatrixLike): Input features.
            y (MatrixLike | ArrayLike): True labels.

        Returns:
            dict: Dictionary with evaluation metrics.
        """
        check_is_fitted(self, "is_fitted_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, (pd.Series, list)):
            y = np.array(y)
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
        }
