# File: modules/preprocess/pca_feature_reduction.py

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_feature_reduction(df: pd.DataFrame, n_components=0.95, columns: list = None) -> pd.DataFrame:
    """
    DataFrame内の数値データにPCAを適用し、特徴量削減を行います。

    Parameters:
        df (pd.DataFrame): 入力のDataFrame。
        n_components (int or float, default=0.95): 
            - intの場合は、保持する主成分の数を指定。
            - floatの場合（0〜1の間）は、保持する累積説明分散比率の閾値を指定。
        columns (list, optional): PCA対象とするカラムのリスト。
            指定しない場合は、DataFrame内の全ての数値型カラムが対象になります。

    Returns:
        pd.DataFrame: 指定したカラムをPCAによる主成分に置き換えたDataFrame。
    """
    df = df.copy()

    # PCA対象カラムが指定されていない場合、全数値型カラムを対象とする
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    # 対象カラムのデータを標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[columns])

    # PCAを実行
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # PCAで得られた主成分に対してカラム名を作成
    pca_columns = [f'pca_{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, index=df.index, columns=pca_columns)

    # 元の対象カラムを削除し、PCA特徴量と連結
    df.drop(columns=columns, inplace=True)
    df = pd.concat([df, df_pca], axis=1)

    return df
