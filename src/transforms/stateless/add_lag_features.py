import pandas as pd

def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    # MultiIndex 構造の確認
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 2:
        raise ValueError("DataFrame のカラムは MultiIndex（2 階層以上）である必要があります。")

    # 第１階層のシンボル一覧を取得
    syms = df.columns.get_level_values(0).unique()

    # 新しいラグ列をタプルキー→Series の dict に格納
    new_cols = {}
    for sym in syms:
        # sym に属する二階層目のカラム名一覧
        subcols = df[sym].columns
        for sub in subcols:
            series = df[(sym, sub)]
            for lag in lags:
                key = (sym, f"{sub}_lag{lag}")
                new_cols[key] = series.shift(lag)

    # dict から DataFrame を作成し、MultiIndex を復元
    lag_df = pd.DataFrame(new_cols, index=df.index)
    lag_df.columns = pd.MultiIndex.from_tuples(lag_df.columns)

    # 元の df と結合、列名をソートして返す
    result = pd.concat([df, lag_df], axis=1)
    result = result.sort_index(axis=1)
    return result
