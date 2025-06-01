import pandas as pd

def add_rolling_statistics(df: pd.DataFrame, windows: list = [5, 10, 20]) -> pd.DataFrame:

    # MultiIndex 構造の確認
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
        raise ValueError("DataFrame のカラムは MultiIndex（2 階層以上）である必要があります。")

    # 第１階層のラベル（例: 'BTCUSDT', 'ETHUSDT', ...）
    syms = df.columns.get_level_values(0).unique()

    new_cols = {}
    for sym in syms:
        # sym に属する二階層目のカラム名一覧
        subcols = df[sym].columns
        for sub in subcols:
            series = df[(sym, sub)]
            for w in windows:
                new_cols[(sym, f"{sub}_roll_mean_{w}")] = series.rolling(window=w).mean()
                new_cols[(sym, f"{sub}_roll_std_{w}")]  = series.rolling(window=w).std()
                new_cols[(sym, f"{sub}_roll_min_{w}")]  = series.rolling(window=w).min()
                new_cols[(sym, f"{sub}_roll_max_{w}")]  = series.rolling(window=w).max()

    # 新しい列群を DataFrame 化して MultiIndex を復元
    feat_df = pd.DataFrame(new_cols, index=df.index)
    feat_df.columns = pd.MultiIndex.from_tuples(feat_df.columns)

    # 元の df と結合し、列を整列して返す
    result = pd.concat([df, feat_df], axis=1)
    result = result.sort_index(axis=1)
    return result
