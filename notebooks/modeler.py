import pandas as pd
from factor_analyzer.factor_analyzer import FactorAnalyzer

import numpy as np
import vizer


def efa(df: pd.DataFrame, n_factors: int | None = None):
    """
    Performs exploratory factor analysis (EFA) with maximum likelihood fitting and
    Oblimin rotation, which is the preferred method with correlated factors.

    Args:
        df (pd.DataFrame): The DataFrame containing the scale features/variables.
        n_factors (int | None, optional): The number of factors to extract.
            If None, the number of factors is not pre-specified. Defaults to None.

    Returns:
        FactorAnalyzer: The fitted FactorAnalyzer object.
    """
    if n_factors:
        fa = FactorAnalyzer(method="ml", rotation="Oblimin", n_factors=n_factors)
    else:
        fa = FactorAnalyzer(method="ml", rotation="Oblimin")

    fa.fit(df)  # Not using the reverse scored item

    return fa


def parallel_analysis(
    df: pd.DataFrame,
    K: int = 10,
    print_eigenvalues: bool = False,
    show_scree_plot: bool = False,
    max_scree_factors: int = 20,
) -> int:
    """
    Performs parallel analysis to determine the number of factors to retain.

    Args:
        df (pd.DataFrame): The input data for factor analysis.
        K (int): The number of iterations for the random data generation.
        print_eigenvalues (bool): Whether to print eigenvalues.
        show_scree_plot (bool): Whether to show a scree plot for the parallel analysis.
        max_scree_factors (int) Max number of factors to show in the scree plot.

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
        sum_factor_eigens = sum_factor_eigens + f_ev  # common factor eigens

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

    if print_eigenvalues:
        print("Factor eigenvalues for random data:\n", avg_factor_eigens)
        print("Factor eigenvalues for real data:\n", data_ev)

    print(
        f"Parallel analysis suggests that the number of factors = {suggested_factors}",
    )

    if show_scree_plot:
        vizer.scree_parallel_analysis(max_scree_factors, avg_factor_eigens, data_ev)

    return suggested_factors


def sort_loadings_with_names(loadings: np.ndarray, item_names: list):
    """
    Sorts factor loadings by absolute value within each factor,
    keeping the item names aligned with the sorted loadings.

    Args:
        loadings (np.ndarray): The NumPy array containing the factor loadings.
        item_names (list): A list of item names corresponding to the rows
                            of the loadings array.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The sorted loadings array.
            - list: The list of item names sorted according to the loadings.
    """

    sorted_loadings = np.zeros_like(loadings)
    sorted_item_names = []  # List to store sorted item names
    for factor_idx in range(loadings.shape[1]):
        factor_loadings = loadings[:, factor_idx]
        sorted_indices = np.argsort(np.abs(factor_loadings))[::-1]
        sorted_loadings[:, factor_idx] = factor_loadings[sorted_indices]
        sorted_item_names.append(
            [item_names[i] for i in sorted_indices]
        )  # Sort item names
    return sorted_loadings, sorted_item_names


def factor_loadings_table(
    loadings: np.ndarray, item_names: list, factor_names: list
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame table of sorted factor loadings.

    Args:
        loadings (np.ndarray): The NumPy array containing the factor loadings.
        item_names (list): A list of lists, where each inner list contains
            the item names for a specific factor, sorted according to the loadings.
        factor_names (list): A list of factor names corresponding to the
            columns of the loadings array.

    Returns:
        pd.DataFrame: A DataFrame containing the sorted factor loadings with
            item names and factor names.
    """

    # If there's only one factor, ensure item_names is a list of lists
    if loadings.shape[1] == 1 and not isinstance(item_names[0], list):
        item_names = [item_names]

    loadings_dict = {}
    for factor_idx, factor_name in enumerate(factor_names):
        loadings_dict[factor_name] = pd.Series(
            loadings[:, factor_idx], index=item_names[factor_idx]
        )

    return pd.DataFrame(loadings_dict)


def get_items_with_low_loadings(
    loadings: np.ndarray, item_names: list, threshold: float
) -> list:
    """
    Gets items that have a loading below the threshold for all factors.

    Args:
        loadings (np.ndarray): The NumPy array containing the factor loadings.
        item_names (list): A list of item names corresponding to the rows
            of the loadings array.
        threshold (float): The threshold for which you only want features
            with loadings above this value.

    Returns:
        list: A list of item names with loadings below the threshold on all factors.
    """

    low_loading_items = []
    for i, item_name in enumerate(item_names):
        if all(abs(loading) < threshold for loading in loadings[i, :]):
            low_loading_items.append(item_name)
    return low_loading_items
