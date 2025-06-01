import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def apply_min_max_scaler(df: pd.DataFrame, columns=None):
    """
    指定したカラムまたは数値カラム全体に MinMaxScaler を適用して正規化する関数
    
    Parameters:
        df (pd.DataFrame): 入力データフレーム
        columns (list or None): 正規化を適用するカラム名のリスト。Noneの場合、数値型の全カラムを対象とする。
    
    Returns:
        pd.DataFrame: 正規化後のデータフレーム
        MinMaxScaler: 適用した scaler（後で逆変換などに利用可能）
    """
    # 正規化対象カラムの決定（指定がなければ全数値カラム）
    if columns is None:
        columns = df.select_dtypes(include='number').columns.tolist()
    
    scaler = MinMaxScaler()
    # fit_transform で scaler の学習と変換を同時に実施
    df[columns] = scaler.fit_transform(df[columns])
    
    return df, scaler