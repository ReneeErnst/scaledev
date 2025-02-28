from typing import Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from factor_analyzer.factor_analyzer import FactorAnalyzer
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.libqsturng import psturng  # Import for Games-Howell
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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


def one_way_anova(df: pd.DataFrame, group_col: str, dv_col: str) -> tuple:
    """
    Performs a one-way ANOVA and Tukey's HSD post-hoc test.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        group_col (str): The name of the column containing the group labels.
        dv_col (str): The name of the column containing the values for the dependent variable.

    Returns:
        tuple: A tuple containing the ANOVA table (statsmodels ANOVA object) and the Tukey's HSD results (statsmodels TukeyHSDResults object).
    """
    # Perform one-way ANOVA
    model = ols(f"{dv_col} ~ C({group_col})", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=1)

    # Perform Tukey's HSD post-hoc test
    tukey_results = pairwise_tukeyhsd(df[dv_col], df[group_col])

    return anova_table, tukey_results


def welch_anova_and_games_howell(
    df: pd.DataFrame, group_col: str, dv_col: str
) -> tuple:
    """
    Performs Welch's ANOVA and Games-Howell post-hoc test.

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        group_col (str): The name of the column containing the group labels.
        dv_col (str): The name of the column containing the values for the dependent variable.

    Returns:
        tuple: A tuple containing the Welch's ANOVA results (statsmodels results object)
               and the Games-Howell results (DataFrame). Returns None for Games-Howell
               if Welch's ANOVA is not significant.
    """

    # Perform Welch's ANOVA
    welch_anova_result = sms.anova_oneway(df[dv_col], df[group_col], use_var="unequal")

    # Perform Games-Howell post-hoc test *only if* Welch's ANOVA is significant
    if welch_anova_result.pvalue < 0.05:
        games_howell_results = games_howell(df, group_col, dv_col)  # Simplified call
        return welch_anova_result, games_howell_results
    else:
        return welch_anova_result, None


def games_howell(df: pd.DataFrame, group_col: str, dv_col: str):
    """
    Performs Games-Howell post-hoc test.  Simplified and corrected.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        group_col (str): Name of the group column.
        dv_col (str): Name of the dependent variable column.

    Returns:
        pd.DataFrame: Games-Howell results.
    """

    groups = df[group_col].unique()
    results = []

    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            # Correctly extract the data for each group as NumPy arrays
            group1 = df.loc[df[group_col] == groups[i], dv_col].to_numpy()
            group2 = df.loc[df[group_col] == groups[j], dv_col].to_numpy()

            n1 = len(group1)
            n2 = len(group2)
            mean1 = np.mean(group1)
            mean2 = np.mean(group2)
            var1 = np.var(group1, ddof=1)  # Sample variance (ddof=1 is crucial)
            var2 = np.var(group2, ddof=1)

            # Calculate t-statistic and degrees of freedom
            t_stat = (mean1 - mean2) / np.sqrt((var1 / n1) + (var2 / n2))
            df_num = (var1 / n1 + var2 / n2) ** 2
            df_denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            df_val = df_num / df_denom

            # Calculate p-value using psturng (studentized range distribution)
            p_val = psturng(np.abs(t_stat) * np.sqrt(2), len(groups), df_val)

            results.append([groups[i], groups[j], mean1 - mean2, t_stat, df_val, p_val])

    results_df = pd.DataFrame(
        results, columns=["Group1", "Group2", "Mean Diff", "t", "df", "p-value"]
    )
    return results_df


def one_way_manova(df: pd.DataFrame, group_col: str, dv_cols: list) -> tuple:
    """
    Performs a one-way MANOVA and follow-up analyses (if significant).

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        group_col (str): The name of the column containing the group labels
            (independent variable, e.g., ethnicity).
        dv_cols (list): A *list* of strings, where each string is the name of a
            column containing a dependent variable (e.g., subscale scores, total score).

    Returns:
        tuple:  Returns a tuple.  The first element is the MANOVA results object. If the
            MANOVA is significant, the second element is a dictionary of ANOVA tables
            (one for each DV), and the third element is a dictionary of Tukey HSD
            results (one for each DV). If the MANOVA is not significant, the second
            and third elements are None.
    """
    # Check for sufficient data:  Crucial for MANOVA
    min_group_size = df.groupby(group_col).size().min()
    if min_group_size <= len(dv_cols):
        raise ValueError(
            f"MANOVA not suitable. The smallest group size ({min_group_size}) must be greater than the number of dependent variables ({len(dv_cols)})."
        )
    # Construct the formula string.
    formula = f"{' + '.join(dv_cols)} ~ C({group_col})"

    # Fit the MANOVA model
    maov = MANOVA.from_formula(formula, data=df)
    manova_results = maov.mv_test()

    # Follow-up Analyses (ONLY if MANOVA is significant)
    anova_tables = {}
    tukey_results = {}

    # Check overall MANOVA significance.  We'll use Pillai's Trace for robustness.
    if (
        manova_results.results[f"C({group_col})"]["stat"].iloc[0, 4] < 0.05
    ):  # Pillai's Trace, p-value
        for dv in dv_cols:
            # One-way ANOVA for each DV
            model = smf.ols(f"{dv} ~ C({group_col})", data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=1)
            anova_tables[dv] = anova_table

            # Tukey's HSD post-hoc test for each DV
            tukey_result = pairwise_tukeyhsd(df[dv], df[group_col])
            tukey_results[dv] = tukey_result
        return manova_results, anova_tables, tukey_results

    else:
        # Return None for ANOVA and Tukey results if MANOVA is not significant.
        return manova_results, None, None


def one_way_manova_games_howell(
    df: pd.DataFrame, group_col: str, dv_cols: list, alpha: float = 0.05
) -> tuple:
    """
    Performs a one-way MANOVA and follow-up analyses (if significant).

    Args:
        df (pd.DataFrame): The pandas DataFrame containing the data.
        group_col (str): The name of the column containing the group labels.
        dv_cols (list): List of dependent variable column names.
        alpha (float): Significance level (default: 0.05).

    Returns:
        tuple: (MANOVA results,
                Dictionary of Welch's ANOVA results (or None),
                Dictionary of Games-Howell results (or None)).
    """

    # Check for sufficient data
    min_group_size = df.groupby(group_col).size().min()
    if min_group_size <= len(dv_cols):
        raise ValueError(
            f"MANOVA not suitable. Smallest group size ({min_group_size}) must be > DVs ({len(dv_cols)})."
        )

    # 1. MANOVA
    formula = f"{' + '.join(dv_cols)} ~ C({group_col})"
    maov = MANOVA.from_formula(formula, data=df)
    manova_results = maov.mv_test()

    # 2. Follow-up Analyses (ONLY if MANOVA is significant)
    welch_anova_tables = {}
    games_howell_results = {}

    # Check overall MANOVA significance (Pillai's Trace)
    if manova_results.results[f"C({group_col})"]["stat"].iloc[0, 4] < alpha:
        for dv in dv_cols:
            # Welch's ANOVA for each DV
            welch_anova_result = sms.anova_oneway(
                df[dv], df[group_col], use_var="unequal"
            )
            welch_anova_tables[dv] = welch_anova_result

            # Games-Howell post-hoc test *only if* Welch's ANOVA is significant
            if welch_anova_result.pvalue < alpha:
                games_howell_results[dv] = games_howell(df, group_col, dv)
            else:
                games_howell_results[dv] = None  # Explicitly set to None

        return manova_results, welch_anova_tables, games_howell_results
    else:
        return manova_results, None, None
