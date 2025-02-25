import pandas as pd
import numpy as np

def triple_barrier_labels(df, symbol, take_profit=0.002, stop_loss=-0.002, time_limit=60, fee=0.00055):
    """
    トリプルバリアの目的変数を計算します。手数料を考慮します。

    Parameters:
    - df (pd.DataFrame): データフレーム。インデックスはタイムスタンプでソートされている必要があります。
    - take_profit (float): 利益目標（例：0.0025は0.25%）
    - stop_loss (float): 損失目標（例：-0.001は-0.1%）
    - time_limit (int): 時間制限（分単位）
    - fee (float): 取引手数料（例：0.001は0.1%）

    Returns:
    - pd.DataFrame: 元のデータフレームに 'target' 列を追加したもの
    """
    df = df.copy()
    df['target'] = 1  # デフォルトは1（時間制限で到達しなかった場合）


    #close_prices = df[f'{target_symbols}_Close'].values
    close_prices = df[f'{symbol}_Close'].values

    timestamps = df.index
    n = len(df)

    for i in range(n):
        current_price = close_prices[i]
        current_time = timestamps[i]

        # 時間制限の終了時刻
        end_time = current_time + pd.Timedelta(minutes=time_limit)

        # 検索範囲の終了インデックス
        j = i + 1
        while j < n and timestamps[j] <= end_time:
            price = close_prices[j]
            return_pct = (price - current_price) / current_price

            if return_pct >= take_profit:
                # Take Profit達成時のネット利益を計算
                net_profit = return_pct - 2 * fee
                if net_profit > 0:
                    df.at[df.index[i], 'target'] = 2  # Take Profit達成
                    break
            elif return_pct <= stop_loss:
                # Stop Loss達成時のネット損失を計算
                net_loss = return_pct - 2 * fee
                # 必要に応じて損失の閾値を調整
                df.at[df.index[i], 'target'] = 0  # Stop Loss達成
                break
            j += 1
        # ループを抜けても達成していない場合は1のまま

    return df
