import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def highlight_corr(val: float):
    """Highlight columns with correlations above 0.6 in green, and less than 0.3 in orange.

    Args:
        val (float): each value in the matrix

    Returns:
        atring: Input to use for color on each matrix item.
    """
    if abs(val) > 0.6 and val != 1.0:
        color = "green"
    elif abs(val) < 0.3:
        color = "blue"
    else:
        color = ""
    return f"background-color: {color}"


def corr_matrix(df: pd.DataFrame, cols: list):
    corr = df[cols].corr()
    corr_style = corr.style.map(highlight_corr)
    return corr_style


def scree_plot(ev: np.ndarray):
    plt.scatter(
    range(1, 14), ev
)  # Only show the first 13, as a starting point from the Kaiser-Gutterman Criterion
    plt.plot(range(1, 14), ev)
    plt.title("Scree Plot")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid()
    plt.show()
    