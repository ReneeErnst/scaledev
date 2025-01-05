import pandas as pd
from factor_analyzer.factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as plt
import numpy as np


def efa(df: pd.DataFrame):
    fa = FactorAnalyzer(
        rotation="Oblimin"
    )  # Oblique rotation since we have correlated factors
    fa.fit(df)  # Not using the reverse scored item

    return fa


def parallel_analysis(
    df: pd.DataFrame,
    K: int = 10,
    printEigenvalues: bool = False,
    max_scree_factors: int = 20,
) -> int:
    """
    Performs parallel analysis to determine the number of factors to retain.

    Args:
        df (pd.DataFrame): The input data for factor analysis.
        K (int): The number of iterations for the random data generation.
        printEigenvalues (bool): Whether to print eigenvalues.
        max_scree_factors (int) Max number of factors to show in the scree plot

    Returns:
        int: The number of suggested factors.
    """
    n_rows, n_features = df.shape

    # Create arrays to store eigenvalues
    sum_factor_eigens = np.empty(n_features)

    # Generate random data and calculate eigenvalues
    for _ in range(K):
        fa = efa(np.random.normal(size=(n_rows, n_features)))
        f_ev = fa.get_eigenvalues()[1]
        sum_factor_eigens = (
            sum_factor_eigens + f_ev
        )  # common factor eigens

    # Average eigenvalues over iterations
    avg_factor_eigens = sum_factor_eigens / K

    # Perform factor analysis on the actual data
    fa.fit(df)
    data_ev = fa.get_eigenvalues()[1]
    
    # Determine the number of factors to retain - only those where the 
    # difference is positive AND both values are positive to begin
    # with. Prevents issues subtracting negative numbers resulting in positives. 
    # Mask for positive values in both arrays
    
    # Calculate the difference
    diff = data_ev - avg_factor_eigens

    # Apply the positive mask to the differences
    masked_diff = diff[(data_ev > 0) & (avg_factor_eigens > 0)]
    
    suggested_factors = sum(masked_diff > 0)

    if printEigenvalues:
        print("Factor eigenvalues for random data:\n", avg_factor_eigens)
        print("Factor eigenvalues for real data:\n", data_ev)

    print(
        f"Parallel analysis suggests that the number of factors = {suggested_factors}",
    )

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

    return suggested_factors
