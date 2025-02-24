import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names in dataframe.

    Args:
        df (pd.DataFrame): Dataframe to clean columns for.

    Returns:
        pd.DataFrame: Dataframe with cleaned column names.
    """
    # Remove all characters before "("
    df.columns = df.columns.map(lambda x: x.split("(", 1)[1] if "(" in x else x)

    # Remove all characters after ")"
    df.columns = df.columns.map(lambda x: x.split(")", 1)[0] if ")" in x else x)

    # Rename columns to lowercase
    df.columns = df.columns.str.lower().columns = df.columns.str.lower()

    # Remove any rows where all data is missing
    df = df.dropna(how="all")

    return df


def corrected_item_total_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates corrected item-total correlations (item-rest correlations) for a DataFrame.

    Args:
        dataframe: A pandas DataFrame where rows are respondents and columns are items.

    Returns:
        A pandas dataframe containing the corrected item-total correlations for each item.
    """
    correlations = []
    items = []
    for col in df.columns:
        items.append(col)
        other_items = df.drop(col, axis=1)
        total_score = other_items.sum(axis=1)
        correlation = df[col].corr(total_score)
        correlations.append(correlation)

    result_df = pd.DataFrame(
        {"Item": items, "Corrected_Item_Total_Correlation": correlations}
    )

    # Sort by correlation in descending order
    result_df = result_df.sort_values(
        by="Corrected_Item_Total_Correlation", ascending=False
    )

    return result_df


def vif(df: pd.DataFrame) -> pd.DataFrame:
    X = add_constant(df)

    # Calculate VIFs
    df_vif = pd.DataFrame()
    df_vif["feature"] = X.columns
    df_vif["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(len(X.columns))
    ]
    return df_vif


def scale_totals(df: pd.DataFrame, scale_items: list[str]) -> pd.DataFrame:
    """Add subscale (factor) totals and ETS total to dataframe.

    Args:
        df (pd.DataFrame): Dataframe to add the total scale values to.
        scale_items (list[str]): List of scale items - used to filter out non-scale items for total calc.

    Returns:
        pd.DataFrame: DF with the scale totals added.
    """
    # Create the factor and total scores
    df["inclusion_total"] = df[
        ["inclusion1", "inclusion2", "inclusion3", "inclusion4", "inclusion5"]
    ].sum(axis=1)
    df["presence_total"] = df[
        ["presence1", "presence2", "presence3", "presence4", "presence5", "presence6"]
    ].sum(axis=1)
    df["embod_total"] = df[["embod1", "embod2", "embod3", "embod4", "embod5"]].sum(
        axis=1
    )
    df["wonder_total"] = df[["wonder1", "wonder2", "wonder3", "wonder4"]].sum(axis=1)
    df["ets_total"] = df[scale_items].sum(axis=1)

    return df
