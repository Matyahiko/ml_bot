import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_volatility_targets_15m(df, symbol, past_window, future_window):
    """
    15分足データに対して、過去の実現ボラティリティと未来の実現ボラティリティ（ターゲット変数）を計算します。
    
    Parameters:
    - df (pd.DataFrame): 15分足のデータフレーム。インデックスはタイムスタンプ、'{symbol}_Close' 列が存在している前提。
    - symbol (str): シンボル名（例: 'AAPL'）
    - past_window (int): 過去何本の15分足データで実現ボラティリティを計算するか（例: 84本）
    - future_window (int): 未来何本の15分足データで実現ボラティリティを計算するか（例: 28本）
    
    Returns:
    - pd.DataFrame: 元のデータフレームに、未来の実現ボラティリティを示す 'target_vol' 列を追加したもの
    """
    df = df.copy()
    
    # 終値からリターンを計算（%変化）
    df['return'] = df[f'{symbol}_Close'].pct_change()
    
    # 過去の実現ボラティリティ：直近past_window本のリターンの標準偏差に√(ウィンドウサイズ)を掛ける
    df[f'{symbol}_realized_vol'] = df['return'].rolling(window=past_window).std() * np.sqrt(past_window)
    
    # 未来の実現ボラティリティを計算するため、各時点で未来のfuture_window本のリターンの標準偏差を求める
    # ※シフトして未来データを参照
    future_returns = df['return'].shift(-1)
    future_vol = future_returns[::-1].rolling(window=future_window).std() * np.sqrt(future_window)
    df['target_vol'] = future_vol[::-1]
    
    # 一時的に作成した'return'列は不要なため削除
    df.drop(columns=['return'], inplace=True)
    
    # 計算できなかった先頭・末尾の行を削除（NaNを含む）
    df.dropna(subset=[f'{symbol}_realized_vol', 'target_vol'], inplace=True)
    
    # ターゲットボラティリティの分布をヒストグラムで描画し、画像として保存
    plt.figure(figsize=(10, 6))
    plt.hist(df['target_vol'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Target Volatility Distribution')
    plt.xlabel('Target Volatility')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("plots/kline/target_vol_distribution.png")
    plt.close()
    
    # --- realized_vol の分布をプロットしたい場合は、以下のコメントアウトを解除してください ---
    # plt.figure(figsize=(10, 6))
    # plt.hist(df[f'{symbol}_realized_vol'], bins=50, alpha=0.7, color='green', edgecolor='black')
    # plt.title('Realized Volatility Distribution')
    # plt.xlabel('Realized Volatility')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.savefig("realized_vol_distribution.png")
    # plt.close()
    
    # 不要になった realized_vol 列を削除
    df.drop(columns=[f'{symbol}_realized_vol'], inplace=True)
    
    return df
