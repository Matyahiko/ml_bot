import pandas as pd

def detect_choch(df, lookback=5):
    """
    CHoCH（Change of Character）を検出します。
    
    Parameters:
    - df (pd.DataFrame): データフレーム。'High' および 'Low' カラムが必要。
    - lookback (int): 過去のバー数を参照する期間。
    
    Returns:
    - pd.Series: CHoCHが発生したポイントをTrueで示すシリーズ。
    """
    # トレンドの変化を検出するために移動平均を使用
    df['ma_short'] = df['Close'].rolling(window=lookback).mean()
    df['ma_long'] = df['Close'].rolling(window=lookback*2).mean()
    
    # 短期移動平均が長期移動平均を上抜けまたは下抜けした場合
    df['CHoCH'] = (df['ma_short'] > df['ma_long']) & (df['ma_short'].shift(1) <= df['ma_long'].shift(1)) | \
                 (df['ma_short'] < df['ma_long']) & (df['ma_short'].shift(1) >= df['ma_long'].shift(1))
    
    df.drop(columns=['ma_short', 'ma_long'], inplace=True)
    df.dropna(inplace=True)
    return df