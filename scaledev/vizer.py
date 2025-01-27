import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats


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

    sns.heatmap(
        loadings,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 8},
        xticklabels=factor_names,
        yticklabels=item_names,
    )

    plt.title("Factor Loadings Heatmap")
    plt.xlabel("Factors")
    plt.ylabel("Items")
    plt.show()


def check_normality(df: pd.DataFrame, dist_plot: bool = True, qq_plot: bool = True):
    """
    Checks the normality assumption of variables in a DataFrame using
    descriptive statistics, statistical tests, and visual methods.

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        dist_plot (bool): Whether to display distribution plots (histograms).
        qq_plot (bool): Whether to display Q-Q plots.
    """

    for col in df.columns:
        print(f"*** Normality Checks for: {col} ***")

        # Descriptive Statistics
        print("\nDescriptive Statistics:")
        print(df[col].describe())
        print(f"Skewness: {df[col].skew()}")
        print(f"Kurtosis: {df[col].kurtosis()}")

        # Kolmogorov-Smirnov Test (preferred since our sample size is > 50)
        # Need to standardize the data for KS test
        ks_test = stats.kstest((df[col] - df[col].mean()) / df[col].std(), "norm")
        print(
            f"Kolmogorov-Smirnov Test: Statistic={ks_test.statistic}, p-value={ks_test.pvalue}"
        )

        # Visual Checks (if requested)
        if dist_plot or qq_plot:
            fig, ax = plt.subplots(
                1, 2 if dist_plot and qq_plot else 1, figsize=(12, 4)
            )

            if dist_plot:
                # Distribution Plot (Histogram)
                sns.histplot(df[col], kde=True, ax=ax[0] if qq_plot else ax)
                ax[0].set_title(f"Distribution Plot of {col}")

            if qq_plot:
                # Q-Q Plot
                stats.probplot(df[col], dist="norm", plot=ax[1] if dist_plot else ax)
                ax[1].set_title(f"Q-Q Plot of {col}")

            plt.tight_layout()
            plt.show()

        print("-" * 40, "\n")
