import pandas as pd

def flatten_columns(df: pd.DataFrame,
                    sep: str = "_") -> pd.DataFrame:
    """
    MultiIndex カラムを 'level1_level2_...' の単一名に変換する。
    既に単層カラムならそのまま。
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            sep.join(map(str, col)).rstrip(sep)  # ('BTC','Open') → 'BTC_Open'
            if isinstance(col, tuple) else str(col)
            for col in df.columns
        ]
    return df