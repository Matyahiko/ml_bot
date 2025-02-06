# data_fetch.py

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
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df

def fetch_multiple_bybit_data(symbol='ETHUSDT', interval='15', limit=200, loops=3):
    """
    指定したシンボルの過去データをループで取得してまとめる関数
    """
    interval_int = int(interval)
    interval_delta = pd.to_timedelta(interval_int, unit='m')
    total_delta = interval_delta * limit
    end_time = pd.Timestamp.utcnow()
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

    return combined_df
