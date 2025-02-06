import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from tqdm import tqdm  


# Bybitからヒストリカルデータを取得する関数（修正済み）
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
    # v5 APIのlistは[open_time, open, high, low, close, volume, turnover]
    df.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Turnover']
    # タイムスタンプをUTCタイムゾーン付きに変換
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.astype(float)
    df.sort_index(inplace=True)
    return df

#FIXME: 時間が飛ぶバグがあるので修正が必要
def accumulate_data(symbol='BTCUSDT', interval='15', limit=200, 
                    start_date='2022-01-01', file_name='raw_data/bybit_15m_data.csv'):
    # 保存ファイルが存在するかチェック
    file_exists = os.path.exists(file_name)
    
    if not file_exists:
        # 初回実行時は start_date から現在時刻までを取得
        start_time = pd.Timestamp(start_date, tz='UTC')
    else:
        # 既存ファイルがある場合は最後のタイムスタンプ以降から取得
        existing_df = pd.read_csv(file_name, parse_dates=['timestamp'], index_col='timestamp')
        # タイムスタンプがタイムゾーン情報を持っているか確認
        if existing_df.index.tz is None:
            # タイムゾーン情報がなければUTCを設定
            existing_df.index = existing_df.index.tz_localize('UTC')
        else:
            # タイムゾーンをUTCに統一
            existing_df.index = existing_df.index.tz_convert('UTC')
        # 最終レコードの時刻
        last_timestamp = existing_df.index.max()
        # 最終取得時刻以降から取得を開始
        start_time = last_timestamp
    
    # 修正箇所: pd.Timestamp.now(tz='UTC') を使用してタイムゾーン情報を持つタイムスタンプを取得
    end_time = pd.Timestamp.now(tz='UTC')
    
    # interval, limitから1回あたりの取得可能期間を計算
    interval_int = int(interval)
    interval_delta = pd.to_timedelta(interval_int, unit='m')
    total_delta = interval_delta * limit  # 200本分の期間
    
    # 総イテレーション数を計算
    total_iterations = ((end_time - start_time) // total_delta) + 1
    
    all_data = []
    
    current_start = start_time

    # tqdmで進捗バーを追加
    with tqdm(total=total_iterations, desc=f"{symbol} データ取得進行状況", unit="バッチ") as pbar:
        while current_start < end_time:
            current_end = current_start + total_delta
            
            # current_end が現在時刻(end_time)より先行く場合はend_timeで止める
            if current_end > end_time:
                current_end = end_time
            
            try:
                df = fetch_bybit_data(symbol=symbol, interval=interval, limit=limit,
                                      start_time=current_start, end_time=current_end)
            except ValueError as e:
                print(f"{symbol}: データ取得中にエラーが発生しました: {e}")
                break
            
            if df.empty:
                # これ以上取れない場合は終了
                print(f"{symbol}: これ以上データが取得できません。終了します。")
                break
            
            all_data.append(df)
            # 取得終了時刻はこのdfの最後のインデックス(=最大timestamp)
            last_fetched_time = df.index.max()
            
            # 次の取得開始時刻は、最後に取れたローソクの時間 + interval_delta
            # (ただしAPIのKLINE取得時、この方法で連続性を維持)
            current_start = last_fetched_time + interval_delta
            
            # レートリミット回避
            time.sleep(0.5)
            
            # 進捗バーを更新
            pbar.update(1)
    
    if not all_data:
        # 新たなデータがなければ終了
        print(f"{symbol}: 新規データはありませんでした。")
        return
    
    new_data = pd.concat(all_data)
    new_data.sort_index(inplace=True)
    new_data.drop_duplicates(inplace=True)
    
    # 既存ファイルがあればマージ
    if file_exists:
        combined_df = pd.concat([existing_df, new_data])
        combined_df.drop_duplicates(inplace=True)
        combined_df.sort_index(inplace=True)
    else:
        combined_df = new_data
    
    # CSV保存（タイムゾーン情報を保持するためISOフォーマットで保存）
    combined_df.to_csv(file_name, date_format='%Y-%m-%dT%H:%M:%SZ')
    print(f"{symbol}: {len(new_data)}件の新規データを追加し、合計{len(combined_df)}件のデータを{file_name}に保存しました。")

if __name__ == "__main__":
    # 収集したいシンボルのリストを定義
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT']  
    
    # データ保存ディレクトリを確認・作成
    output_dir = 'raw_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 各シンボルについてデータを収集
    for symbol in symbols:
        # ファイル名をシンボルごとに変更
        file_name = os.path.join(output_dir, f'bybit_{symbol}_15m_data.csv')
        accumulate_data(symbol=symbol, interval='15', limit=200, 
                       start_date='2022-01-01', file_name=file_name)
