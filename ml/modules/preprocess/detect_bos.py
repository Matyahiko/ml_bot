import pandas as pd

def detect_bos(df, lookback=5):
    """
    BOS（Break of Structure）を検出します。
    
    Parameters:
    - df (pd.DataFrame): データフレーム。'High' および 'Low' カラムが必要。
    - lookback (int): 過去のバー数を参照する期間。
    
    Returns:
    - pd.Series: BOSが発生したポイントをTrueで示すシリーズ。
    """
    highs = df['High']
    lows = df['Low']
    
    # ローカルの高値と安値を計算
    df['local_high'] = highs.rolling(window=lookback, center=True).max()
    df['local_low'] = lows.rolling(window=lookback, center=True).min()
    
    # 現在の高値が過去の高値をブレイク
    df['BOS'] = highs > df['local_high'].shift(1)
    
    # 現在の安値が過去の安値をブレイク
    df['BOS'] = df['BOS'] | (lows < df['local_low'].shift(1))
    
    dorop_columns = ['local_high', 'local_low']
    df.dropna(inplace=True)
    return df