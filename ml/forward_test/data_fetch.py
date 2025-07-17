import os
import requests
import pandas as pd
import time
from datetime import datetime
from rich import print

def fetch_bybit_data(symbol='BTCUSDT', interval='15', limit=200, start_time=None, end_time=None):
    """
    BybitのKLineデータを取得する関数
    """
    url = 'https://api.bybit.com/v5/market/kline'
    params = {
        'category': 'linear',
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    if start_time:
        params['start'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['end'] = int(end_time.timestamp() * 1000)
        
    response = requests.get(url, params=params)
    data = response.json()
    if 'result' not in data or 'list' not in data['result']:
        raise ValueError("APIレスポンスに必要なデータが含まれていません。レスポンス内容: {}".format(data))
    
    df = pd.DataFrame(data['result']['list'])
    # v5 API のリストは [open_time, open, high, low, close, volume, turnover]
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    # タイムスタンプはUTCとして扱う
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df

def fetch_multiple_bybit_data(symbol='ETHUSDT', interval='15', limit=200, loops=3):
    """
    指定したシンボルの過去データをループで取得してまとめる関数  
    ※BTCUSDTの場合は、取得した市場データにオンチェーンデータ（Blockchain.com API：BTC専用）を統合します。
    """
    interval_int = int(interval)
    interval_delta = pd.to_timedelta(interval_int, unit='m')
    total_delta = interval_delta * limit
    # tz付きの現在時刻を直接取得
    end_time = pd.Timestamp.now(tz='UTC')
    all_data = []

    for i in range(loops):
        start_time = end_time - total_delta
        print(f"Fetching loop {i+1}/{loops} for {symbol}")
        df = fetch_bybit_data(
            symbol=symbol,
            interval=interval,
            limit=limit,
            start_time=start_time,
            end_time=end_time
        )
        if df.empty:
            print("これ以上データが取得できません。終了します。")
            break

        all_data.append(df)
        end_time = start_time
        combined_df = pd.concat(all_data).drop_duplicates()
        combined_df.sort_index(inplace=True)
        
        print(f"Total data points for {symbol}: {len(combined_df)}")
        time.sleep(0.5)  

    # BTCUSDTの場合のみ、オンチェーンデータを統合（※Blockchain.com API は BTC のみ対応）
    if symbol.upper() == 'BTCUSDT':
        metrics = ['n-unique-addresses', 'market-price', 'n-transactions', 'estimated-transaction-volume']
        combined_df = merge_onchain_data(combined_df, symbol='BTC', onchain_metrics=metrics)
    
    return combined_df

# --------------------
# オンチェーンデータ取得用関数群（Blockchain.com Charts API）
# --------------------

def fetch_blockchain_onchain_data(metric='n-unique-addresses', start_date='2022-01-01', end_date=None):
    """
    Blockchain.com Charts API を用いてオンチェーンデータを取得します。
    ※Blockchain.com API は BTC 向けのデータのみ提供しています。
    
    metric: 取得する指標（例: 'n-unique-addresses'：ユニークアドレス数）
    """
    start_dt = pd.Timestamp(start_date, tz='UTC')
    if end_date is None:
        end_dt = pd.Timestamp.now(tz='UTC')
    else:
        end_dt = pd.Timestamp(end_date, tz='UTC')
    days_diff = (end_dt - start_dt).days
    timespan = f"{days_diff}days"
    
    url = f"https://api.blockchain.info/charts/{metric}"
    params = {
        'timespan': timespan,
        'format': 'json',
        'cors': 'true'
    }
    response = requests.get(url, params=params)
    data = response.json()
    if 'values' not in data:
        raise ValueError(f"Blockchain.com APIエラー: {data}")
    df = pd.DataFrame(data['values'])
    df.rename(columns={'x': 'timestamp', 'y': metric}, inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df.loc[(df.index >= start_dt) & (df.index <= end_dt)]
    return df

def fetch_blockchain_multi_onchain_data(metrics, start_date, end_date):
    """
    複数の指標データを一括取得し、インデックスで結合します。
    """
    dfs = []
    for metric in metrics:
        try:
            df_metric = fetch_blockchain_onchain_data(metric=metric, start_date=start_date, end_date=end_date)
            dfs.append(df_metric)
        except Exception as e:
            print(f"チャート {metric} の取得に失敗しました: {e}")
    if dfs:
        multi_df = pd.concat(dfs, axis=1)
    else:
        multi_df = pd.DataFrame()
    return multi_df

def merge_onchain_data(market_df, symbol, onchain_metrics=['n-unique-addresses', 'market-price', 'n-transactions', 'estimated-transaction-volume']):
    """
    市場データとオンチェーンデータを統合します。  
    Blockchain.com API は BTC のみ対応しているため、引数 symbol が 'BTC' でない場合は統合をスキップします。
    """
    if symbol != 'BTC':
        print(f"{symbol}: Blockchain.com API は BTC のみ対応のため、オンチェーンデータ統合をスキップします。")
        return market_df
    
    start_date = market_df.index.min().date().isoformat()
    end_date = market_df.index.max().date().isoformat()
    
    try:
        onchain_df = fetch_blockchain_multi_onchain_data(onchain_metrics, start_date, end_date)
    except Exception as e:
        print(f"{symbol}: オンチェーンデータの取得に失敗しました: {e}")
        return market_df
    
    # 市場データの足（例：15分足）に合わせるため、オンチェーンデータを前方充填でリサンプリング
    onchain_resampled = onchain_df.resample('15min').ffill()
    merged_df = market_df.join(onchain_resampled, how='left')
    merged_df.ffill(inplace=True)
    return merged_df

# --------------------
# __main__ (テスト用)
# --------------------
if __name__ == "__main__":
    # 例: BTCUSDTの場合、マーケットデータにオンチェーンデータも統合されます。
    sym = 'BTCUSDT'
    config = {
        'interval': '15',
        'limit': 200,
        'loops': 3
    }
    df_sym = fetch_multiple_bybit_data(
        symbol=sym,
        interval=config['interval'],
        limit=config['limit'],
        loops=config['loops']
    )
    print(df_sym)
