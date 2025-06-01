

"""
【概要】
・各データセット（train, test）の bybit_BTCUSDT_15m_data_{train|test}.pkl を読み込み、
  学習済みCatBoostマルチタスクモデル（LongReturn, ShortReturn, LongVolatility, LongVolatility以外も出力）による予測を実施。
・予測結果（LongReturn_predictions, ShortReturn_predictions, LongVolatility_predictions）を用い、
  ロングエントリー（予測値が各閾値を超えた場合）によるシンプルなトレードシミュレーションを実施。
・各エントリーシグナルごとのトレード結果（利益）がプラスなら meta_label=1、そうでなければ0 として二値ラベル付与します。
・全データセットの結果を結合し、meta_label を含むデータセット（pickle形式）として保存します。
"""

import os
import sys
import datetime
import pandas as pd
import pickle
import numpy as np
import math

import backtrader as bt
from catboost import CatBoostRegressor

# --- データ処理モジュールのインポート ---
from modules.rnn_base.rnn_data_process import TimeSeriesDataset

# =============================
# ① CatBoost 推論用関数
# =============================
def catboost_predict(df, model_path, window_size=405, stride=1):
    """
    学習済みCatBoostマルチタスクモデルで予測を実施する関数

    このモデルはマルチタスクモデルであり、以下の6つの予測値を出力します:
      - LongReturn, ShortReturn, LongVolatility, ShortVolatility, LongMaxDD, ShortMaxDD

    Args:
        df (DataFrame): 入力データ
        model_path (str): 学習済みCatBoostモデルのパス
        window_size (int): ウィンドウサイズ
        stride (int): ストライド

    Returns:
        df_pred (DataFrame): ウィンドウ末尾時刻と各予測結果の DataFrame
    """
    # 予測用データセット作成（ターゲット不要）
    dataset = TimeSeriesDataset(
        predict_df=df,
        target_column=None,
        window_size=window_size,
        stride=stride
    )

    # 各ウィンドウの特徴量をフラット化
    features_list = []
    for i in range(len(dataset)):
        sample = dataset[i]  # sample: shape=(window_size, feature_dim)
        features_list.append(sample.numpy().reshape(-1))
    X_test = np.stack(features_list)

    # CatBoostモデルの読み込み
    print("[CatBoost 推論] モデル読み込み:", model_path)
    model = CatBoostRegressor()
    model.load_model(model_path)

    # 予測実施（出力は (N, 6) の配列を想定）
    print("[CatBoost 推論] 予測実施中...")
    preds = model.predict(X_test)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 6)

    # 出力カラム名の定義
    columns = [
        "LongReturn_predictions", "ShortReturn_predictions",
        "LongVolatility_predictions", "ShortVolatility_predictions",
        "LongMaxDD_predictions", "ShortMaxDD_predictions"
    ]

    # ウィンドウの末尾時刻をインデックスとして取得
    ws = window_size
    st = stride
    window_indices = [df.index[i + ws - 1] for i in range(0, len(df) - ws, st)]
    if len(window_indices) != preds.shape[0]:
        print(f"[CatBoost 推論] 警告: ウィンドウ数({len(window_indices)})と予測数({preds.shape[0]})が一致しません。")

    df_pred = pd.DataFrame(preds, index=window_indices, columns=columns)
    return df_pred

# =============================
# ② Backtrader 用データフィード
# =============================
class BybitCSV(bt.feeds.PandasData):
    """
    Backtrader用カスタムデータフィード
    ※BTCUSDTのOHLCVと CatBoost 予測結果を使用する前提
    """
    lines = ('LongReturn_predictions', 'ShortReturn_predictions',
             'LongVolatility_predictions', 'ShortVolatility_predictions',
             'LongMaxDD_predictions', 'ShortMaxDD_predictions',)
    params = (
        ('datetime', None),
        ('open', 'BTCUSDT_Open'),
        ('high', 'BTCUSDT_High'),
        ('low', 'BTCUSDT_Low'),
        ('close', 'BTCUSDT_Close'),
        ('volume', 'BTCUSDT_Volume'),
        ('LongReturn_predictions', 'LongReturn_predictions'),
        ('ShortReturn_predictions', 'ShortReturn_predictions'),
        ('LongVolatility_predictions', 'LongVolatility_predictions'),
        ('ShortVolatility_predictions', 'ShortVolatility_predictions'),
        ('LongMaxDD_predictions', 'LongMaxDD_predictions'),
        ('ShortMaxDD_predictions', 'ShortMaxDD_predictions'),
        ('openinterest', -1),
    )

# =============================
# ③ CatBoost予測に基づく ロング戦略（ラベル付け用）
# =============================
class CatBoostPredictedLongStrategy(bt.Strategy):
    """
    ロング戦略 (CatBoost予測に基づくエントリー):
      - エントリー条件:
          現在バーの予測値（LongReturn_predictions, ShortReturn_predictions, LongVolatility_predictions）
          がそれぞれ設定した閾値を超えた場合にエントリー。
      - エグジット:
          エントリー後、ストップロス／テイクプロフィットが発動するか、保有期間が終了した場合にエグジット。
      - 各トレード終了時、エントリー時点の利益がプラスなら meta_label=1、そうでなければ0 として記録します。
    """
    params = (
        ('hold_period', 1),             # 保有期間（バー数）
        ('stop_loss', 0.02),            # ストップロス幅（例:2%）
        ('take_profit', 0.05),          # テイクプロフィット幅（例:5%）
        ('long_return_threshold', 0.0), # LongReturn予測の閾値
        ('short_return_threshold', 0.0),# ShortReturn予測の閾値
        ('long_vol_threshold', 0.0),    # LongVolatility予測の閾値
    )

    def __init__(self):
        # 現在保有中のトレード管理用リスト
        self.open_trades = []
        # トレード終了時の情報を記録（entry_time, profit）
        self.trades_info = []

    def next(self):
        current_dt = self.data.datetime.datetime(0)
        o = self.data.open[0]
        h = self.data.high[0]
        l = self.data.low[0]
        c = self.data.close[0]

        ########################################
        # 1. 既存トレードのエグジット処理
        ########################################
        trades_to_remove = []
        for trade in self.open_trades:
            # エントリー直後バーはスキップ
            if trade.get('just_entered', False):
                trade['just_entered'] = False
                continue
            if trade.get('exited', False):
                continue

            SL = trade['SL']
            TP = trade['TP']
            exit_price = None

            # ロングの場合、open→high→low→close の順でストップ／テイクチェック
            if o >= TP:
                exit_price = o
            elif o <= SL:
                exit_price = o
            elif h >= TP:
                exit_price = TP
            elif l <= SL:
                exit_price = SL

            if exit_price is not None:
                trade['prices'].append(exit_price)
                trade['exited'] = True
                entry_price = trade['entry_price']
                profit = np.log(exit_price / entry_price)
                # トレード終了情報を記録（利益が正なら1、負なら0）
                meta_label = 1 if profit > 0 else 0
                self.trades_info.append({'entry_time': trade['entry_dt'], 'profit': profit, 'meta_label': meta_label})
                trades_to_remove.append(trade)
            else:
                trade['prices'].append(c)
                trade['remaining'] -= 1
                if trade['remaining'] <= 0:
                    exit_price = c
                    trade['exited'] = True
                    entry_price = trade['entry_price']
                    profit = np.log(exit_price / entry_price)
                    meta_label = 1 if profit > 0 else 0
                    self.trades_info.append({'entry_time': trade['entry_dt'], 'profit': profit, 'meta_label': meta_label})
                    trades_to_remove.append(trade)
        self.open_trades = [t for t in self.open_trades if not t.get('exited', False)]

        ########################################
        # 2. 新規エントリー条件のチェック（ロングのみ）
        ########################################
        pred_long_return = self.data.LongReturn_predictions[0] if hasattr(self.data, 'LongReturn_predictions') else None
        pred_short_return = self.data.ShortReturn_predictions[0] if hasattr(self.data, 'ShortReturn_predictions') else None
        pred_long_vol = self.data.LongVolatility_predictions[0] if hasattr(self.data, 'LongVolatility_predictions') else None

        if (pred_long_return is not None and pred_short_return is not None and pred_long_vol is not None):
            if (pred_long_return > self.p.long_return_threshold and
                pred_short_return > self.p.short_return_threshold and
                pred_long_vol > self.p.long_vol_threshold):
                # エントリー条件を満たす場合、ロングエントリー
                entry_price = o
                trade = {
                    'entry_dt': current_dt,
                    'entry_price': entry_price,
                    'remaining': self.p.hold_period,
                    'prices': [entry_price],
                    'SL': entry_price * (1 - self.p.stop_loss),
                    'TP': entry_price * (1 + self.p.take_profit),
                    'exited': False,
                    'just_entered': True,
                }
                self.open_trades.append(trade)

# =============================
# ④ 各データセットの処理を実行する関数
# =============================
def process_file(file_tag):
    """
    指定ファイル（train または test）の pkl ファイルを読み込み、
    CatBoostモデルによる予測と、バックテストシミュレーションを実施し、
    各エントリー時刻におけるトレード結果に基づいて二値ラベル（meta_label）を付与して返す。
    """
    filename = f"storage/kline/bybit_BTCUSDT_15m_data_{file_tag}.pkl"
    print(f"\n[{file_tag.upper()}] ファイル読み込み: {filename}")

    try:
        df = pd.read_pickle(filename)
        print(f"[{file_tag.upper()}] DataFrame shape: {df.shape}")
    except Exception as e:
        print(f"[{file_tag.upper()}] データ読み込み失敗: {e}")
        return pd.DataFrame()

    df = df.dropna()  # 必要に応じて欠損行除去

    # --- CatBoostによる予測 ---
    catboost_model_path = "models/catboost/catboost_model_final.cbm"
    print(f"[{file_tag.upper()}] CatBoost予測開始 (model: {catboost_model_path})")
    df_pred = catboost_predict(
        df,
        model_path=catboost_model_path,
        window_size=2,
        stride=1
    )

    # マージ：ウィンドウ末尾の時刻に対して予測結果を設定
    for col in df_pred.columns:
        df.loc[df_pred.index, col] = df_pred[col]

    # 欠損値は 0 で補完
    for col in df_pred.columns:
        df[col] = df[col].fillna(0).astype(float)

    # --- バックテストによるシグナル毎トレードシミュレーション ---
    print(f"[{file_tag.upper()}] Backtraderでバックテスト開始")
    cerebro = bt.Cerebro(stdstats=False)
    data_feed = BybitCSV(dataname=df)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(CatBoostPredictedLongStrategy)
    cerebro.broker.setcash(1000000)
    results = cerebro.run()
    strat = results[0]

    # --- 各トレード結果から meta_label を算出 ---
    df["meta_label"] = 0
    for trade in strat.trades_info:
        entry_time = trade['entry_time']
        meta_label = trade['meta_label']
        if entry_time in df.index:
            df.at[entry_time, "meta_label"] = meta_label
        else:
            # インデックスが一致しない場合、最も近い時刻に対応する処理も検討可能
            pass

    print(f"[{file_tag.upper()}] トレード件数: {len(strat.trades_info)}")
    return df

# =============================
# ⑤ メイン処理：各データセットを処理して結合
# =============================
def main():
    data_types = ["train", "test"]
    df_list = []

    for file_tag in data_types:
        print(f"\n===== {file_tag.upper()} データ処理開始 =====")
        df_file = process_file(file_tag)
        df_list.append(df_file)

    # train, test を結合（インデックス重複に注意）
    df_combined = pd.concat(df_list).sort_index()
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')]
    
    output_filename = "storage/kline/bybit_BTCUSDT_15m_data_with_meta_labels.pkl"
    df_combined.to_pickle(output_filename)
    print(f"\n[完了] meta label を付与したデータセットを保存: {output_filename}")

if __name__ == "__main__":
    main()
