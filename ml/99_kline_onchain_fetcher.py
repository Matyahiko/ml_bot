import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

# Bybitからヒストリカルデータを取得する関数
def fetch_bybit_data(symbol='BTCUSDT', interval='15', limit=200, start_time=None, end_time=None):
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
    # v5 APIのlistは [open_time, open, high, low, close, volume, turnover]
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df

# Blockchain.com Charts APIから1つの指標データを取得する関数
def fetch_blockchain_onchain_data(metric='n-unique-addresses', start_date='2022-01-01', end_date=None):
    """
    Blockchain.com Charts API を用いてオンチェーンデータを取得します。
    ※ Blockchain.com API は BTC 向けのデータのみ提供しています。
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

# 複数の指標データを一括取得し、インデックスで結合する関数
def fetch_blockchain_multi_onchain_data(metrics, start_date, end_date):
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

# 市場データとオンチェーンデータを統合する関数
def merge_onchain_data(market_df, symbol, onchain_metrics=['n-unique-addresses', 'market-price', 'n-transactions', 'estimated-transaction-volume']):
    # Blockchain.com API は BTC のみ対応しているため、BTC以外の場合は統合をスキップ
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
    
    # 日次データを15分足に合わせるため前方充填
    onchain_15min = onchain_df.resample('15min').ffill()
    merged_df = market_df.join(onchain_15min, how='left')
    merged_df.ffill(inplace=True)
    return merged_df

# Bybitデータの蓄積＋オンチェーンデータ統合を行う関数
def accumulate_data(symbol='BTCUSDT', interval='15', limit=200, 
                    start_date='2022-01-01', file_name='raw_data/bybit_15m_with_onchain.csv', include_onchain=False):
    file_exists = os.path.exists(file_name)
    
    if not file_exists:
        start_time = pd.Timestamp(start_date, tz='UTC')
    else:
        existing_df = pd.read_csv(file_name, parse_dates=['timestamp'], index_col='timestamp')
        if existing_df.index.tz is None:
            existing_df.index = existing_df.index.tz_localize('UTC')
        else:
            existing_df.index = existing_df.index.tz_convert('UTC')
        last_timestamp = existing_df.index.max()
        start_time = last_timestamp
    
    end_time = pd.Timestamp.now(tz='UTC')
    
    interval_int = int(interval)
    interval_delta = pd.to_timedelta(interval_int, unit='m')
    total_delta = interval_delta * limit
    
    total_iterations = ((end_time - start_time) // total_delta) + 1
    all_data = []
    current_start = start_time

    with tqdm(total=total_iterations, desc=f"{symbol} データ取得進行状況", unit="バッチ") as pbar:
        while current_start < end_time:
            current_end = current_start + total_delta
            if current_end > end_time:
                current_end = end_time
            
            try:
                df = fetch_bybit_data(symbol=symbol, interval=interval, limit=limit,
                                      start_time=current_start, end_time=current_end)
            except ValueError as e:
                print(f"{symbol}: データ取得中にエラーが発生しました: {e}")
                break
            
            if df.empty:
                print(f"{symbol}: これ以上データが取得できません。終了します。")
                break
            
            all_data.append(df)
            last_fetched_time = df.index.max()
            current_start = last_fetched_time + interval_delta
            time.sleep(0.5)
            pbar.update(1)
    
    if not all_data:
        print(f"{symbol}: 新規データはありませんでした。")
        return
    
    new_data = pd.concat(all_data)
    new_data.sort_index(inplace=True)
    new_data.drop_duplicates(inplace=True)
    
    if file_exists:
        combined_df = pd.concat([existing_df, new_data])
        combined_df.drop_duplicates(inplace=True)
        combined_df.sort_index(inplace=True)
    else:
        combined_df = new_data
    
    combined_df.to_csv(file_name, date_format='%Y-%m-%dT%H:%M:%SZ')
    print(f"{symbol}: {len(new_data)}件の新規データを追加し、合計{len(combined_df)}件のデータを{file_name}に保存しました。")
    
    # BTCUSDTの場合のみ、オンチェーンデータ（BTCのみ）を統合
    if include_onchain and symbol == 'BTCUSDT':
        metrics = ['n-unique-addresses', 'market-price', 'n-transactions', 'estimated-transaction-volume']
        # Blockchain.com API では coin シンボルは 'BTC' として扱う
        merged_df = merge_onchain_data(combined_df, symbol='BTC', onchain_metrics=metrics)
        merged_file_name = file_name.replace('.csv', '_merged.csv')
        merged_df.to_csv(merged_file_name, date_format='%Y-%m-%dT%H:%M:%SZ')
        print(f"{symbol}: On-chainデータを統合したファイルを {merged_file_name} に保存しました。")

if __name__ == "__main__":
    # klineデータは 'BTCUSDT', 'ETHUSDT', 'XRPUSDT' の3種類を取得します。
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    output_dir = 'raw_data'
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol in symbols:
        # BTCUSDT の場合のみオンチェーンデータ統合を行う
        include_onchain = True if symbol == 'BTCUSDT' else False
        file_name = os.path.join(output_dir, f'bybit_{symbol}_15m.csv')
        accumulate_data(symbol=symbol, interval='15', limit=200, 
                        start_date='2022-01-01', file_name=file_name, include_onchain=include_onchain)
