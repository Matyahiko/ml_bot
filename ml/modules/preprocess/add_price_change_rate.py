import pandas as pd
import numpy as np

def add_price_change_rate(df: pd.DataFrame, symbols: str, n: int = 1) -> pd.DataFrame:
    """
    各シンボルについて、nステップ先の対数リターンを計算し、新たなカラムとして追加する関数。
    
    計算式:
        対数リターン = ln( nステップ先の終値 / 現在の終値 )
    
    Parameters:
        df (pd.DataFrame): 入力データフレーム（時系列データ）
        symbols (str): シンボル。各シンボルの終値は "{シンボル}_Close" というカラム名である前提。
        n (int): 何ステップ先の対数リターンを計算するか（デフォルトは1）
        
    Returns:
        pd.DataFrame: 新たなカラムが追加されたデータフレーム
    """
    df = df.copy()
    price_col = f"{symbols}_Close"
    target_col = "log_return"
    df[target_col] = np.log(df[price_col].shift(-n) / df[price_col])
    df[[target_col]].to_csv('storage/kline/temp/df_labeled.csv')
    return df