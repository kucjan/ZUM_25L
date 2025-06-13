from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.validation import check_is_fitted

from config.constants import (
    GENERATOR_PARAMS,
    MIN_UNIQUE_NUM_VALUES,
    OUTLIER_DIRECTION_PROB,
    OUTLIER_RATIO,
    SEED,
)
from config.types import ArrayLike, MatrixLike

rng = np.random.default_rng(seed=SEED)


class OneClassWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model: BaseEstimator,
        outliers_mode: Literal["default", "mahalanobis"] = "default",
        outlier_ratio: float = OUTLIER_RATIO,
        **kwargs,
    ) -> None:
        """Initializes the OneClassWrapper for one-class classification with synthetic outliers.

        Args:
            model (BaseEstimator): A scikit-learn compatible binary classification model.
            outliers_mode (Literal["default", "mahalanobis"], optional): The method to generate synthetic outliers.
                                                                        'default' for simple range extension.
                                                                        'mahalanobis' for Mahalanobis-distance based generation.
                                                                        Defaults to "default".
            outlier_ratio (float, optional): The ratio of synthetic outliers to generate relative to the original data size.
                                             Defaults to OUTLIER_RATIO.
            **kwargs: Additional parameters specific to the chosen 'outliers_mode'.
                      These parameters will override defaults in `GENERATOR_PARAMS`.
                      For `outliers_mode="default"`, parameters include:
                          - `extend_factor` (float): Factor for extending outlier value range.
                          - `shrink_factor` (float): Factor for shrinking outlier value range (for the 'other' side).
                      For `outliers_mode="mahalanobis"`, parameters include:
                          - `extend_factor` (float): Factor to push points outward from data center.
                          - `noise_factor` (float): Overall intensity multiplier for added noise.
                          - `extreme_ratio` (float): Proportion of outliers to make extreme.
                          - `extreme_extend_scaler` (float): Multiplier for pushing extreme points further.
                          - `extreme_noise_scaler` (float): Multiplier for heavier noise on extreme points.
        """
        self.model = model
        self.outliers_mode = outliers_mode
        self.outlier_ratio = outlier_ratio
        self.gen_params = kwargs if kwargs else GENERATOR_PARAMS[outliers_mode]

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

    def _generate_mahalanobis_outliers(
        self,
        X: pd.DataFrame,
        n_outliers: int,
    ) -> np.ndarray:
        """Generates multivariate outliers based on Mahalanobis distance.

        Args:
            X: Input features (numerical DataFrame).
            n_outliers: Number of outlier samples to generate.

        Returns:
            np.ndarray: Array of generated outlier values.
        """
        X_values = X.to_numpy()
        mean = X_values.mean(axis=0)
        cov = np.cov(X_values, rowvar=False)

        # Generating base outliers using Mahalanobis scaling
        points = rng.multivariate_normal(mean=mean, cov=cov, size=n_outliers)
        inv_cov = np.linalg.inv(cov)
        centered = points - mean
        distances = np.sqrt(np.sum(centered @ inv_cov * centered, axis=1))
        median_dist = np.median(distances)

        outliers = mean + (points - mean) * (
            self.gen_params["extend_factor"] * median_dist / distances[:, np.newaxis]
        )

        # Additional noise
        noise = rng.normal(
            scale=np.sqrt(np.diag(cov)) / self.gen_params["noise_factor"],
            size=outliers.shape,
        )
        outliers += noise

        # Including extreme outliers
        n_extreme = int(n_outliers * self.gen_params["extreme_ratio"])
        if n_extreme > 0:
            extreme_idx = np.argpartition(distances, -n_extreme)[-n_extreme:]

            outliers[extreme_idx] = mean + (points[extreme_idx] - mean) * (
                self.gen_params["extreme_extend_scaler"]
                * self.gen_params["extend_factor"]
                * median_dist
                / distances[extreme_idx, np.newaxis]
            )

            # Adding heavier noise to extremes
            outliers[extreme_idx] += rng.normal(
                scale=self.gen_params["extreme_noise_scaler"]
                * np.sqrt(np.diag(cov))
                / self.gen_params["noise_factor"],
                size=(n_extreme, X_values.shape[1]),
            )

        return outliers

    def _generate_numerical_outliers(
        self,
        X: pd.DataFrame,
        n_outliers: int,
    ) -> pd.DataFrame:
        """Generates numerical outlier rows for DataFrame X by perturbing existing rows.

        Args:
            X: Numerical DataFrame.
            n_outliers: Number of outlier samples to generate.

        Returns:
            DataFrame of generated outlier rows.
        """
        generated_outliers_data = []

        col_min_vals = X.min()
        col_max_vals = X.max()
        col_ranges = col_max_vals - col_min_vals

        num_cols = X.shape[1]

        min_outlier_cols = 1

        for _ in range(n_outliers):
            base_row = X.iloc[rng.integers(0, len(X))].copy()
            generated_row = base_row.copy()

            n_outlier_cols_for_this_row = rng.integers(min_outlier_cols, num_cols)
            outlier_col_indices = rng.choice(
                num_cols, n_outlier_cols_for_this_row, replace=False
            )
            outlier_cols = X.columns[outlier_col_indices]

            for col_name in X.columns:
                col_min = col_min_vals[col_name]
                col_max = col_max_vals[col_name]
                col_range = col_ranges[col_name]

                extend_scaler = col_range * self.gen_params["extend_factor"]
                shrink_scaler = col_range * self.gen_params["shrink_factor"]

                if col_name in outlier_cols:
                    if rng.random() < OUTLIER_DIRECTION_PROB:
                        generated_row[col_name] = rng.uniform(
                            low=col_max - shrink_scaler,
                            high=col_max + extend_scaler,
                        )
                    else:
                        generated_row[col_name] = rng.uniform(
                            low=col_min - extend_scaler,
                            high=col_min + shrink_scaler,
                        )

                else:
                    generated_row[col_name] = rng.uniform(col_min, col_max)
            generated_outliers_data.append(generated_row)

        return pd.DataFrame(generated_outliers_data, columns=X.columns)

    def _generate_categorical_outliers(
        self, col: pd.Series, n_outliers: int
    ) -> np.ndarray:
        """Generates categorical outlier values by sampling existing unique values.

        Args:
            col (pd.Series): Categorical column from the dataset.
            n_outliers (int): Number of outlier samples to generate.

        Returns:
            np.ndarray: Array of sampled outlier values.
        """
        unique_vals = col.unique()
        all_options = unique_vals
        return rng.choice(all_options, size=n_outliers)

    def _generate_outliers(self, X: pd.DataFrame) -> pd.DataFrame:
        n_outliers = int(len(X) * self.outlier_ratio)
        col_types = self._determine_column_types(X)

        num_cols = [col for col in X.columns if col_types[col] == "numerical"]
        cat_cols = [col for col in X.columns if col_types[col] == "categorical"]

        numerical_outliers = (
            self._generate_numerical_outliers(X[num_cols], n_outliers)
            if self.outliers_mode == "default"
            else self._generate_mahalanobis_outliers(X, n_outliers)
        )
        outlier_df = pd.DataFrame(numerical_outliers, columns=num_cols)

        for col in cat_cols:
            outlier_df[col] = self._generate_categorical_outliers(X[col], n_outliers)

        return outlier_df

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
        """Calculates the F1-score of the model.

        Args:
            X (MatrixLike): Input features.
            y (MatrixLike | ArrayLike): True labels.

        Returns:
            float: F1-score.
        """
        return self.evaluate(X, y)["f1"]

    def evaluate(self, X: MatrixLike, y: MatrixLike | ArrayLike) -> dict:
        """Evaluates the model using precision, recall, F1-score, and false positive rate.

        Includes confusion matrix components: TP, FP, FN, TN.

        Args:
            X (MatrixLike): Input features.
            y (MatrixLike | ArrayLike): True labels.

        Returns:
            dict: Dictionary with evaluation metrics and confusion matrix components.
        """
        check_is_fitted(self, "is_fitted_")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if isinstance(y, (pd.Series, list)):
            y = np.array(y)

        y_pred = self.predict(X)

        tn, fp, _, _ = confusion_matrix(y, y_pred, labels=[1, 0]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return {
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "fpr": fpr,
        }
