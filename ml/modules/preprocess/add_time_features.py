import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrameのタイムスタンプ（indexまたは'timestamp'列）から基本的な時間情報と周期性を捉える正弦波・余弦波特徴量を追加する関数

    Args:
        df (pd.DataFrame): 時系列データを含むDataFrame。indexがDatetimeIndexであるか、もしくは'timestamp'列が必要です。

    Returns:
        pd.DataFrame: 時間ベースの特徴量が追加されたDataFrame
    """
    # タイムスタンプ情報を取得（indexがDatetimeIndexでない場合は 'timestamp' 列を利用）
    if isinstance(df.index, pd.DatetimeIndex):
        time_series = df.index
    elif 'timestamp' in df.columns:
        time_series = pd.to_datetime(df['timestamp'])
    else:
        raise ValueError("日時情報が見つかりません。DataFrameのindexがDatetimeIndexか、'timestamp'列が必要です。")
    
    # 基本的な時間成分の追加
    df['year'] = time_series.year
    df['month'] = time_series.month
    df['day'] = time_series.day
    df['hour'] = time_series.hour
    df['minute'] = time_series.minute
    df['day_of_week'] = time_series.dayofweek  # 0=月曜日, 6=日曜日

    # 周期性を捉えるための正弦波・余弦波特徴量
    df['hour_sin'] = np.sin(2 * np.pi * time_series.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * time_series.hour / 24)
    
    df['dow_sin'] = np.sin(2 * np.pi * time_series.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * time_series.dayofweek / 7)
    
    df['month_sin'] = np.sin(2 * np.pi * time_series.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * time_series.month / 12)
    
    # 内部断片化解消のためにコピーして返す
    return df.copy()