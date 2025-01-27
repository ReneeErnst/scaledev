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
