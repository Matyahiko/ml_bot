import pandas as pd

def detect_fvg(df, threshold=0.001):
    """
    FVG（Fair Value Gap）を検出します。
    
    Parameters:
    - df (pd.DataFrame): データフレーム。'High' および 'Low' カラムが必要。
    - threshold (float): ギャップを判定する閾値。
    
    Returns:
    - pd.DataFrame: FVGを検出した結果を含むデータフレーム。
    """
    df = df.copy()
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    
    # ギャップの検出: 前のバーの高値と現在のバーの安値の間にギャップがあるか
    df['FVG'] = (df['Low'] > df['prev_high'] * (1 + threshold)) | (df['High'] < df['prev_low'] * (1 - threshold))
    
    return df

def fvg_regression(df):
    """
    FVGが発生した後の価格の回帰を確認します。
    
    Parameters:
    - df (pd.DataFrame): データフレーム。'Close' カラムが必要。
    
    Returns:
    - pd.DataFrame: FVG回帰が確認できたポイントを含むデータフレーム。
    """
    # 簡単な回帰例: FVG発生後一定期間内に価格がギャップを埋めるかを確認
    regression_period = 10  # バー数
    
    df['FVG_regression'] = False
    fvg_indices = df.index[df['FVG']].tolist()
    
    for idx in fvg_indices:
        loc = df.index.get_loc(idx)
        start = loc + 1
        end = start + regression_period
        if end > len(df):
            end = len(df)
        window = df.iloc[start:end]
        
        # ギャップが埋められたかを確認
        if ((window['High'] >= df.at[idx, 'High']) & (window['Low'] <= df.at[idx, 'Low'])).any():
            df.at[idx, 'FVG_regression'] = True
    
    return df
