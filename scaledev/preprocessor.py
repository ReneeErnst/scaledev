import pandas as pd


def clean_columns(df: pd.DataFrame):
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
