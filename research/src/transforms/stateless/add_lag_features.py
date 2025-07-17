import pandas as pd

def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        raise ValueError("この関数はフラットなカラム（1階層）専用です。")

    # timeで始まる列は除外
    target_cols = [col for col in df.columns if not str(col).startswith("time")]

    new_cols = {}
    for col in target_cols:
        for lag in lags:
            new_col = f"{col}_lag{lag}"
            new_cols[new_col] = df[col].shift(lag)

    # 新しいラグ特徴量DataFrame
    lag_df = pd.DataFrame(new_cols, index=df.index)

    # 元データと結合して返す
    result = pd.concat([df, lag_df], axis=1)
    return result