import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def wvf_triple(df, wvf_threshold=70, take_profit=15.0, stop_loss=-10.0, time_limit=60, fee=0.00055, debug=True):
    """
    新しい教師ラベルを付与します。
    
    条件:
    - WVFが高水準（wvf_threshold%以上）
    - BOSまたはCHoCH指標が発生 (BOS==1またはCHoCH==1)
    - FVG回帰が確認できる (FVG_regression==1)
    - これらが揃ったときの価格上昇確率を学習するためのラベル付け
    
    Parameters:
    - df (pd.DataFrame): データフレーム。必要なカラム: 'Close', 'WVF', 'BOS', 'CHoCH', 'FVG_regression'
    - wvf_threshold (float): WVFの高水準を判定する閾値（パーセンテージ）
    - take_profit (float): 利益目標（パーセンテージ）
    - stop_loss (float): 損失目標（パーセンテージ）
    - time_limit (int): 時間制限（分単位）
    - fee (float): 取引手数料（パーセンテージ）
    - debug (bool): デバッグモード。Trueの場合、プロットを生成・一部print
    
    Returns:
    - pd.DataFrame: 元のデータフレームに 'target' 列を追加したもの
    """
    df = df.copy()
    df['target'] = 0  # 初期値は0
    
    # 必要な列があるか確認
    required_cols = ['Close', 'WVF', 'BOS', 'CHoCH', 'FVG_regression']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the DataFrame.")
    
    close_prices = df['Close'].values
    timestamps = df.index
    n = len(df)
    
    # インデックスがDatetimeIndexかどうか確認（時間制限にpd.Timedeltaを使うため）
    if not isinstance(timestamps, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a DateTimeIndex for time_limit logic to work.")
    
    for i in range(n):
        # 条件1: WVFが高水準
        if df.iloc[i]['WVF'] < wvf_threshold:
            continue
        
        # 条件2: BOSまたはCHoCHが発生 (BOS==1またはCHoCH==1)
        if not ((df.iloc[i]['BOS'] == 1) or (df.iloc[i]['CHoCH'] == 1)):
            continue
        
        # 条件3: FVG回帰が確認できる(FVG_regression==1)
        if df.iloc[i]['FVG_regression'] != 1:
            continue
        
        # ここまで来た時点でエントリー条件揃い
        current_price = close_prices[i]
        current_time = timestamps[i]
        
        # 時間制限の終了時刻
        end_time = current_time + pd.Timedelta(minutes=time_limit)
        
        # 時間制限内でTP/SLヒットをチェック
        j = i + 1
        entry_labeled = False
        while j < n and timestamps[j] <= end_time:
            price = close_prices[j]
            return_pct = (price - current_price) / current_price * 100.0  # パーセンテージ
            
            # 利確条件達成
            if return_pct >= take_profit:
                # 手数料考慮後の利益
                net_profit = return_pct - (2 * fee)
                if net_profit > 0:
                    df.at[df.index[i], 'target'] = 1
                else:
                    df.at[df.index[i], 'target'] = 0
                entry_labeled = True
                break
            
            # 損切り条件達成
            if return_pct <= stop_loss:
                # 手数料考慮後の損益（ほぼ損）
                # stop_lossの場合、必ず0でよいかどうかは戦略次第
                net_loss = return_pct - (2 * fee)
                df.at[df.index[i], 'target'] = 0
                entry_labeled = True
                break
            
            j += 1
        
        # jループ抜け後、TPもSLも達成しなかった場合は0のまま
        # 既に初期値0なので特に操作不要
        
        if debug and entry_labeled:
            print(f"Entry at {timestamps[i]} labeled with {df.at[df.index[i],'target']} (TP/SL hit within {time_limit} mins)")
    
    if debug:
        # 教師ラベルの分布をプロット
        plt.figure(figsize=(8, 6))
        sns.countplot(x='target', data=df)
        plt.title('Distribution of Teacher Labels')
        plt.xlabel('Target New')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig('plots/preprocess/teacher_labels_distribution.png')
        plt.close()
    
    return df
