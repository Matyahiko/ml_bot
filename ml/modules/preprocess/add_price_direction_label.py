import pandas as pd

def add_price_direction_label(df: pd.DataFrame, symbol: str, n: int = 1) -> pd.DataFrame:
    """
    指定されたシンボルの終値から、nステップ先の価格変化率を計算し、
    価格の方向を示すラベルを追加する関数です。
    
    計算式:
        価格変化率 = (nステップ先の終値 - 現在の終値) / 現在の終値
        
    価格方向ラベルの付け方:
        - 終値の変化率が正の場合: 1（上昇）
        - 終値の変化率が0または負の場合: 0（下降または変化なし）
        
    Parameters:
        df (pd.DataFrame): 入力の時系列データフレーム
        symbol (str): シンボル名。終値は "{シンボル}_Close" というカラム名である前提。
        n (int): nステップ先の価格変化率を計算するためのステップ数（デフォルトは1）
        
    Returns:
        pd.DataFrame: 価格変化率と価格方向ラベルが追加されたデータフレーム
    """
    df = df.copy()
    price_col = f"{symbol}_Close"
    # nステップ先の価格変化率を計算
    df["simple_return"] = (df[price_col].shift(-n) - df[price_col]) / df[price_col]
    
    # 価格の方向に応じたラベルを作成
    df["direction"] = df["simple_return"].apply(lambda x: 1 if x > 0 else 0)
    
    df.drop(columns=["simple_return"], inplace=True)
    
    # 必要に応じてCSVに保存（パスは適宜変更してください）
    df[["direction"]].to_csv('storage/kline/temp/df_labeled.csv')
    return df
