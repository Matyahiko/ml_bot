import pandas as pd

def add_future_volatility(df: pd.DataFrame, symbols: str, n: int = 1) -> pd.DataFrame:
    """
    各シンボルについて、nステップ先のボラティリティ（将来 n 日間の日次リターンの標準偏差）を計算し、
    新たなカラムとして追加する関数。
    
    計算方法:
      日次リターン = (翌日の終値 - 当日の終値) / 当日の終値
      ボラティリティ = 将来 n 日間の日次リターンの標準偏差
      
    Parameters:
        df (pd.DataFrame): 入力の時系列データフレーム
        symbols (list): シンボルのリスト。各シンボルの終値は "{シンボル}_close" というカラム名である前提。
        n (int): 何ステップ先のボラティリティを計算するか（デフォルトは 1）
        
    Returns:
        pd.DataFrame: 新たなボラティリティカラムが追加されたデータフレーム
    """
    df = df.copy()
    
    price_col = f"{symbols}_Close"
    target_col = f"{symbols}_future_volatility_{n}"
    # 翌日の日次リターンを計算
    daily_return = (df[price_col].shift(-1) - df[price_col]) / df[price_col]
    # 将来 n 日間の日次リターンの標準偏差を計算（順方向のローリング計算のため、データを反転）
    df[target_col] = daily_return.iloc[::-1].rolling(window=n, min_periods=1).std().iloc[::-1]
    return df
