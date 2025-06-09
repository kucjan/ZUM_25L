import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.graph_objs import Figure as PlotlyFigure
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from config.types import ArrayLike, MatrixLike


def plot_confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """
    Plots a confusion matrix for one-class classification results using seaborn.

    Args:
        y_true (ArrayLike): True labels (1 for inliers, 0 for outliers).
        y_pred (ArrayLike): Predicted labels (1 for inliers, 0 for outliers).

    Returns:
        matplotlib.figure.Figure: The confusion matrix figure.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Outlier", "Inlier"],
        yticklabels=["Outlier", "Inlier"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.show()


def pca_visualization(
    X: MatrixLike, y_true: ArrayLike, y_proba: ArrayLike | None = None
) -> PlotlyFigure:
    """
    Visualizes data using PCA (2D) with optional coloring by prediction probabilities.

    Args:
        X (MatrixLike): Feature matrix.
        y_true (ArrayLike): True class labels.
        y_proba (ArrayLike | None, optional): Predicted probabilities of inlier class. Defaults to None.

    Returns:
        PlotlyFigure: An interactive PCA scatter plot.
    """
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    y_true = np.asarray(y_true)

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=y_proba,
            symbol=y_true.astype(str),
            title="Classification Results<br><sup>Color: Inlier Probability | Symbol: True Label</sup>",
            labels={"color": "Inlier Probability", "symbol": "True Class"},
            color_continuous_scale="RdYlGn",
        )
    else:
        fig = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=y_true.astype(str),
            title="Generated Data Structure<br><sup>Color: True Label</sup>",
            labels={"color": "Class"},
        )

    fig.update_layout(
        xaxis_title="PCA Component 1",
        yaxis_title="PCA Component 2",
        hovermode="closest",
    )

    return fig


def plot_inlier_outlier_counts(results: dict) -> PlotlyFigure:
    """
    Plots a grouped bar chart of average inlier and outlier counts per model using Plotly.

    Args:
        results (dict): Dictionary where each key is a model name and values contain "inlier" and "outlier" counts.

    Returns:
        PlotlyFigure: Interactive grouped bar chart with annotations.
    """
    models = list(results.keys())
    inliers = [results[model]["inlier"] for model in models]
    outliers = [results[model]["outlier"] for model in models]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=models,
            y=inliers,
            name="Inliers",
            marker_color="green",
            text=inliers,
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=models,
            y=outliers,
            name="Outliers",
            marker_color="red",
            text=outliers,
            textposition="auto",
        )
    )

    fig.update_layout(
        barmode="group",
        title="Average Inlier and Outlier Counts per Model",
        xaxis_title="Model",
        yaxis_title="Average Count",
        legend_title="Class",
        xaxis_tickangle=45,
    )

    return fig


def plot_evaluation_metrics(results: dict) -> PlotlyFigure:
    """
    Plots grouped bar charts of evaluation metrics (precision, recall, f1, fpr) per model using Plotly.

    Args:
        results (dict): Dictionary where each key is a model name and value is a dictionary with 'metrics'.

    Returns:
        PlotlyFigure: Interactive grouped bar chart of evaluation metrics.
    """
    metrics = ["precision", "recall", "f1", "fpr"]
    models = list(results.keys())

    fig = go.Figure()

    for metric in metrics:
        values = [results[model]["metrics"].get(metric, 0.0) for model in models]
        fig.add_trace(
            go.Bar(
                x=models,
                y=values,
                name=metric,
                text=[f"{v:.2f}" for v in values],
                textposition="auto",
                hovertemplate=f"{metric}: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        title="Evaluation Metrics per Model",
        xaxis_title="Model",
        yaxis_title="Score",
        legend_title="Metric",
        xaxis_tickangle=45,
        yaxis=dict(range=[0, 1.05]),
        template="plotly_white",
    )

    return fig
