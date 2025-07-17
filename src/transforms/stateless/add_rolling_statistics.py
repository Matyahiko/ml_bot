import pandas as pd

def add_rolling_statistics(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:
    # timeで始まる列は除外
    target_cols = [col for col in df.columns if not str(col).startswith("time")]

    # MultiIndexでstack（index, col）
    stacked = df[target_cols].stack()
    # index0: 元のdfのindex、index1: シンボル名

    new_cols = {}
    for w in windows:
        roll_mean = stacked.rolling(window=w, min_periods=1).mean().unstack()
        roll_std  = stacked.rolling(window=w, min_periods=1).std().unstack()
        roll_min  = stacked.rolling(window=w, min_periods=1).min().unstack()
        roll_max  = stacked.rolling(window=w, min_periods=1).max().unstack()

        # 各列に戻してカラム名をつける
        for col in target_cols:
            new_cols[f"{col}_roll_mean_{w}"] = roll_mean[col]
            new_cols[f"{col}_roll_std_{w}"]  = roll_std[col]
            new_cols[f"{col}_roll_min_{w}"]  = roll_min[col]
            new_cols[f"{col}_roll_max_{w}"]  = roll_max[col]

    feat_df = pd.DataFrame(new_cols, index=df.index)
    result = pd.concat([df, feat_df], axis=1)
    return result
