import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wvf(df, length=14):
    """
    Williams VIX Fix (WVF) を計算します。
    
    Parameters:
    - df (pd.DataFrame): データフレーム。'High' および 'Close' カラムが必要。
    - length (int): 計算に使用する期間（デフォルトは14）。
    
    Returns:
    - pd.Series: WVFの値。
    """
    # 過去N期間の最高値を計算
    highest_high = df['High'].rolling(window=length, min_periods=1).max()
    
    # WVFの計算
    wvf = (highest_high - df['Close']) / highest_high * 100
    
    # WVFをデータフレームに追加
    df['WVF'] = wvf
    
    # WVFのプロット
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'], label='Close Price')
    plt.plot(df['WVF'], label='WVF', color='orange')
    plt.legend()
    plt.title('Close Price and Williams VIX Fix')
    plt.savefig('plots/preprocess/wvf.png')
    
    return df
