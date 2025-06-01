import pandas as pd

def add_excess_return(df: pd.DataFrame, symbols: list, n: int = 1, benchmark_col: str = "benchmark_close") -> pd.DataFrame:
    """
    各シンボルについて、nステップ先の超過リターンを計算し、新たなカラムとして追加する関数。
    
    計算方法:
      シンボルのリターン = (nステップ先の終値 - 現在の終値) / 現在の終値
      ベンチマークのリターン = (nステップ先のベンチマーク終値 - 現在のベンチマーク終値) / 現在のベンチマーク終値
      超過リターン = シンボルのリターン - ベンチマークのリターン
      
    Parameters:
        df (pd.DataFrame): 入力の時系列データフレーム
        symbols (list): シンボルのリスト。各シンボルの終値は "{シンボル}_close" というカラム名である前提。
        n (int): 何ステップ先のリターンを計算するか（デフォルトは 1）
        benchmark_col (str): ベンチマークの終値が格納されているカラム名（デフォルトは "benchmark_close"）
        
    Returns:
        pd.DataFrame: 新たな超過リターンカラムが追加されたデータフレーム
    """
    if benchmark_col not in df.columns:
        raise ValueError(f"Benchmark column '{benchmark_col}' not found in DataFrame.")
    
    df = df.copy()
    # ベンチマークの nステップ先リターンを計算
    benchmark_return = (df[benchmark_col].shift(-n) - df[benchmark_col]) / df[benchmark_col]
    
    for sym in symbols:
        asset_col = f"{sym}_Close"
        target_col = f"{sym}_excess_return_{n}"
        asset_return = (df[asset_col].shift(-n) - df[asset_col]) / df[asset_col]
        df[target_col] = asset_return - benchmark_return
    return df
