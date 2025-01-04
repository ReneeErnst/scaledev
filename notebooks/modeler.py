import pandas as pd
from factor_analyzer.factor_analyzer import FactorAnalyzer

def efa(df: pd.DataFrame):
    fa = FactorAnalyzer(
        rotation="Oblimin"
    )  # Oblique rotation since we have correlated factors
    fa.fit(df)  # Not using the reverse scored item

    return fa
