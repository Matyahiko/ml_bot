import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def trend_scan_labels(df, symbol, span=(3, 10), plot=True):
    """
    トレンドスキャンの目的変数を計算し、結果をプロットします。

    Parameters:
    - df (pd.DataFrame): データフレーム。インデックスはタイムスタンプでソートされている必要があります。
    - symbol (str): シンボル名。データフレーム内の '{symbol}_Close' 列を使用します。
    - span (tuple): ルックフォワード期間の範囲（最小、最大）。デフォルトは (3, 10)。
    - plot (bool): ラベリング結果をプロットするかどうか。デフォルトは True。

    Returns:
    - pd.DataFrame: 元のデータフレームに 'target' 列を追加したもの（ただし 't1' 列は削除済み）
    """
    df = df.copy()
    df['target'] = np.nan  # 初期化

    close_prices = df[f'{symbol}_Close']
    timestamps = df.index
    n = len(df)

    # 各列のデータ型を明示的に指定して結果を保持する DataFrame を初期化
    results = pd.DataFrame({
        't1': pd.Series(index=df.index, dtype='datetime64[ns]'),
        'tVal': pd.Series(index=df.index, dtype='float64'),
        'bin': pd.Series(index=df.index, dtype='float64')
    })

    horizons = range(span[0], span[1] + 1)

    for i in range(n):
        current_time = timestamps[i]
        t_values = {}

        for horizon in horizons:
            end_idx = i + horizon
            if end_idx >= n:
                continue
            end_time = timestamps[end_idx]
            segment = close_prices.iloc[i:end_idx + 1]

            if len(segment) < 2:
                continue  # 回帰に必要なデータポイントが不足している場合

            # 線形回帰の準備
            x = np.ones((len(segment), 2))
            x[:, 1] = np.arange(len(segment))
            model = sm.OLS(segment.values, x).fit()
            t_val = model.tvalues[1]
            t_values[end_time] = t_val

        if t_values:
            # 絶対値が最大の t 値に対応する終了時刻を選択
            best_end_time = max(t_values, key=lambda k: abs(t_values[k]))
            best_t_val = t_values[best_end_time]
            bin_label = 1 if best_t_val > 0 else 0  # 正なら 1、負または 0 なら 0

            # best_end_time をタイムゾーンなしにキャスト（datetime64[ns] に合わせる）
            best_end_time_converted = pd.to_datetime(best_end_time).tz_localize(None)
            results.at[current_time, 't1'] = best_end_time_converted
            results.at[current_time, 'tVal'] = best_t_val
            results.at[current_time, 'bin'] = bin_label

    # 'bin' ラベルを df にマージ（NaN は 0 で埋め、整数にキャスト）
    df['target'] = results['bin'].fillna(0).astype(int)
    df['t1'] = results['t1']

    if plot:
        # 結果のプロット
        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df[f'{symbol}_Close'], label='Price')
        colors = {0: 'red', 1: 'blue', 2: 'green'}
        for idx, row in df.dropna(subset=['target', 't1']).iterrows():
            plt.axvspan(idx, row['t1'], color=colors.get(row['target'], 'blue'), alpha=0.3)
        plt.title('Trend Scan Labeling')
        plt.xlabel('Date')
        plt.ylabel('Price')
        # レジェンドの位置を固定することで、loc="best" の自動探索を回避
        plt.legend(loc='upper left')
        plt.savefig('plots/kline/trend_scan_labels.png')

    # 't1' 列は最終的に削除して返す
    return df.drop(columns=['t1'])


def trend_scan_labels_log_return(df, symbol, window=2):
    """
    シンプルな対数リターンベースのラベリング関数。

    Parameters:
    - df (pd.DataFrame): データフレーム。
    - symbol (str): シンボル名。データフレーム内の '{symbol}_Close' 列を使用します。
    - window (int): ルックフォワード期間（タイムステップ数）。デフォルトは 2。

    Returns:
    - pd.DataFrame: 'log_return' 列を追加したデータフレーム。
    """
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
