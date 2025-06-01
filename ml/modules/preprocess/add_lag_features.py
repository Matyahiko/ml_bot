import pandas as pd


def add_lag_features(df: pd.DataFrame, sym: str, lags: list = [1, 2, 3], columns: list = None) -> pd.DataFrame:
    """
    指定されたシンボルのカラムに対してラグ特徴量を一括で追加する関数
    （多数回の挿入によるDataFrameの断片化を防ぐため、pd.concatを使用しています）
    
    Args:
        df (pd.DataFrame): 処理対象のDataFrame
        sym (str): シンボル名（例: 'BTCUSDT'）
        lags (list, optional): 追加するラグの値のリスト。デフォルトは [1, 2, 3]
        columns (list, optional): ラグを追加する対象カラムのリスト。Noneの場合、シンボルに対応する全カラムが対象になる
    
    Returns:
        pd.DataFrame: ラグ特徴量が追加されたDataFrame
    """
    # 対象カラムが指定されていなければ、シンボルのプレフィックスを持つ全カラムを対象とする
    if columns is None:
        columns = [col for col in df.columns if col.startswith(f'{sym}_')]
    
    # 新しいラグ特徴量のSeriesをリストに格納
    lag_features = []
    for col in columns:
        for lag in lags:
            lag_feature_name = f'{col}_lag{lag}'
            lag_features.append(df[col].shift(lag).rename(lag_feature_name))
    
    # 新しいラグ特徴量を一括でDataFrameに変換
    if lag_features:
        lag_features_df = pd.concat(lag_features, axis=1)
        # 元のDataFrameと連結（axis=1で横方向に結合）
        df = pd.concat([df, lag_features_df], axis=1)
    
    # コピーを作成してメモリ内の断片化を解消
    return df.copy()
