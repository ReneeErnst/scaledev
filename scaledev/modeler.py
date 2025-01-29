import pandas as pd
from factor_analyzer.factor_analyzer import FactorAnalyzer
from typing import Tuple
import numpy as np
from scaledev import vizer


def efa(
    df: pd.DataFrame,
    n_factors: int | None = None,
    method: str = "ml",
    rotation="oblimin",
) -> FactorAnalyzer:
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
        fa = FactorAnalyzer(method=method, rotation=rotation, n_factors=n_factors)
    else:
        fa = FactorAnalyzer(method=method, rotation=rotation)

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


def factor_loadings_table(
    loadings: np.ndarray, item_names: pd.Index | list, factor_names: list
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame table of sorted factor loadings.

    Args:
        loadings (np.ndarray): The NumPy array containing the factor loadings.
        item_names (pd.Index | list): A pandas Index or a list of lists,
            where each inner list contains the item names for a specific factor,
            sorted according to the loadings.
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
        # Use item_names directly if it's a pd.Index
        if isinstance(item_names, pd.Index):
            valid_index = item_names
        else:  # Otherwise, use the appropriate list from item_names
            valid_index = item_names[factor_idx]

        loadings_dict[factor_name] = pd.Series(
            loadings[:, factor_idx], index=valid_index
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


def no_low_loadings_solution(
    df: pd.DataFrame, low_loadings: list, n_factors: int
) -> Tuple[pd.DataFrame, FactorAnalyzer]:
    """Re-run efa until we've removed all items with loadings < 0.4.

    Args:
        df (pd.DataFrame): DataFrame for EFA
        low_loadings (list): Low Loadings from the initial EFA
        n_factors (int): Number of factors.
    Returns:
        tuple: The dataframe with all the low loading items removed and the EFA model.
    """
    while len(low_loadings) > 0:
        df = df.drop(columns=low_loadings)
        efa_model = efa(df=df, n_factors=n_factors)

        low_loadings = get_items_with_low_loadings(
            efa_model.loadings_, df.columns, threshold=0.4
        )
        print("Items with low loadings: ")
        print(low_loadings)

        vizer.plot_loadings_heatmap(
            loadings=efa_model.loadings_,
            item_names=df.columns,
            factor_names=[f"Factor {i + 1}" for i in range(efa_model.n_factors)],
        )

    return df.reset_index(drop=True), efa_model


def strongest_loadings(loadings: np.ndarray, item_names: list[str]) -> pd.DataFrame:
    """
    Find which factor each item loads most strongly on

    Args:
        loadings (np.ndarray): The NumPy array containing the factor loadings.
        item_names (list): A list of item names corresponding to the rows
            of the loadings array.

    Returns:
        pd.DataFrame: DataFrame with the items and the strongest factor they load on.
    """
    # Loadings to df
    df_loadings = pd.DataFrame(loadings, index=item_names)

    # Add column names to df_loadings, starting from 1
    num_factors = df_loadings.shape[1]
    df_loadings.columns = [i + 1 for i in range(num_factors)]

    # Find the factor with the highest absolute loading for each item
    strongest_factors = df_loadings.abs().idxmax(axis=1)

    # Create a DataFrame to store the results
    df_item_factors = pd.DataFrame(
        {"item": item_names, "strongest_factor": strongest_factors}
    )

    # Add the actual loadings to the df
    df_item_factors["loading"] = df_item_factors.apply(
        lambda row: df_loadings.loc[row["item"], row["strongest_factor"]], axis=1
    )

    return df_item_factors.sort_values(
        by=["strongest_factor", "loading"], ascending=[True, False]
    )
