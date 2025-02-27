import pandas as pd

def add_rolling_statistics(df: pd.DataFrame, sym: str, windows: list = [5, 10, 20], columns: list = None) -> pd.DataFrame:
    """
    指定されたシンボルのカラムに対してローリング統計量を一括で追加する関数
    （移動平均、移動標準偏差、移動最小値、移動最大値を計算します）
    
    Args:
        df (pd.DataFrame): 処理対象のDataFrame
        sym (str): シンボル名（例: 'BTCUSDT'）
        windows (list, optional): ローリングウィンドウのサイズのリスト。デフォルトは [5, 10, 20]
        columns (list, optional): ローリング統計量を追加する対象カラムのリスト。Noneの場合、シンボルに対応する全カラムが対象になる
        
    Returns:
        pd.DataFrame: ローリング統計量が追加されたDataFrame
    """
    # 対象カラムが指定されていなければ、シンボルのプレフィックスを持つ全カラムを対象とする
    if columns is None:
        columns = [col for col in df.columns if col.startswith(f'{sym}_')]
    
    rolling_features = []
    
    for col in columns:
        for window in windows:
            # 移動平均
            mean_feature_name = f'{col}_roll_mean_{window}'
            rolling_features.append(df[col].rolling(window=window).mean().rename(mean_feature_name))
            
            # 移動標準偏差
            std_feature_name = f'{col}_roll_std_{window}'
            rolling_features.append(df[col].rolling(window=window).std().rename(std_feature_name))
            
            # 移動最小値
            min_feature_name = f'{col}_roll_min_{window}'
            rolling_features.append(df[col].rolling(window=window).min().rename(min_feature_name))
            
            # 移動最大値
            max_feature_name = f'{col}_roll_max_{window}'
            rolling_features.append(df[col].rolling(window=window).max().rename(max_feature_name))
    
    if rolling_features:
        # 各ローリング統計量を一括でDataFrameに結合
        rolling_df = pd.concat(rolling_features, axis=1)
        df = pd.concat([df, rolling_df], axis=1)
    
    # メモリ断片化の対策としてコピーを作成
    return df.copy()
