import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def scree_plot(common_factors_ev: np.ndarray, max_viz: int = 20):
    plt.scatter(
        range(1, max_viz + 1), common_factors_ev[:max_viz]
    )  # Only show the first 13, as a starting point from the Kaiser-Gutterman Criterion
    plt.plot(range(1, max_viz + 1), common_factors_ev[:max_viz])

    plt.title("Scree Plot - Common Factors Eigen values")
    plt.xlabel("Factors")
    plt.ylabel("Eigenvalue")
    plt.grid()

    # Force x-axis to show only integers
    plt.xticks(range(1, max_viz + 1))

    plt.show()


def scree_parallel_analysis(
    max_scree_factors: int, avg_factor_eigens: np.ndarray, data_ev: np.ndarray
):
    # Create scree plot
    plt.figure(figsize=(8, 6))
    plt.plot(
        [0, max_scree_factors + 1], [1, 1], "k--", alpha=0.3
    )  # Line for eigenvalue 1
    plt.plot(
        range(1, max_scree_factors + 1),
        avg_factor_eigens[:max_scree_factors],
        "g",
        label="FA - random",
        alpha=0.4,
    )
    plt.scatter(
        range(1, max_scree_factors + 1), data_ev[:max_scree_factors], c="g", marker="o"
    )
    plt.plot(
        range(1, max_scree_factors + 1),
        data_ev[:max_scree_factors],
        "g",
        label="FA - data",
    )
    plt.title("Parallel Analysis Scree Plots", {"fontsize": 20})
    plt.xlabel("Factors", {"fontsize": 15})
    plt.xticks(
        ticks=range(1, max_scree_factors + 1), labels=range(1, max_scree_factors + 1)
    )
    plt.ylabel("Eigenvalue", {"fontsize": 15})
    plt.legend()
    plt.show()


def plot_loadings_heatmap(
    loadings: np.ndarray, item_names: list[list], factor_names: list
):
    """
    Plots a heatmap of factor loadings.

    Args:
        loadings (np.ndarray): The NumPy array containing the factor loadings.
        item_names (list): A list of lists, where each inner list contains
            the item names for a specific factor, sorted according to the loadings.
        factor_names (list): A list of factor names corresponding to the
            columns of the loadings array.
    """

    plt.figure(figsize=(20, 16))

    # If there's only one factor, ensure item_names is a list of lists
    if loadings.shape[1] == 1 and not isinstance(item_names[0], list):
        item_names = [item_names]  # Wrap item_names in another list

    # Set y-axis labels for each factor separately
    # Combine item names from all factors, BUT remove duplicates
    yticklabels = []
    for factor_item_names in item_names:
        for item_name in factor_item_names:
            if item_name not in yticklabels:  # Check for duplicates
                yticklabels.append(item_name)

    ax = sns.heatmap(
        loadings,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 8},
        xticklabels=factor_names,
        yticklabels=yticklabels,
    )

    plt.title("Factor Loadings Heatmap")
    plt.xlabel("Factors")
    plt.ylabel("Items")
    plt.show()
