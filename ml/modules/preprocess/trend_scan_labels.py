import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def trend_scan_labels_log_return(df, symbol, window=2):
    df = df.copy()
    
    # 2タイムステップ先の終値を取得
    df['close_2steps'] = df[f'{symbol}_Close'].shift(-window)
    
    # 対数リターンを計算: ln(P[t+2] / P[t])
    df['log_return'] = np.log(df['close_2steps'] / df[f'{symbol}_Close'])
    
    # 上昇なら1、下降なら0としてラベル付け（必要なら有効化）
    # df['target_trend'] = np.where(df['log_return'] > 0, 1, 0)
    
    df.drop(columns=['close_2steps'], inplace=True)
    df = df.iloc[:-window]  # シフトによるNaNを除去
    
    # 対数リターンの分布をヒストグラムで描画し、画像として保存
    plt.figure(figsize=(10, 6))
    plt.hist(df['log_return'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Log Return Distribution')
    plt.xlabel('Log Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("plots/kline/log_return_distribution.png")  # 画像ファイルとして保存
    plt.close()
    
    return df
