#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py

【概要】
・各シンボルのデータを取得し、テクニカル指標計算を実施。
・CatBoostマルチタスクモデルにより、LongReturn, ShortReturn, LongVolatility,
  ShortVolatility, LongMaxDD, ShortMaxDD の6指標を予測。
・BacktraderのデータフィードにCatBoost予測結果とOHLCVをセットし、SMAフィルタ付き
  ロング・ショート戦略にて、かつ予測の閾値条件（ロングならLongReturn > 0.001、短期ならShortReturn < -0.001）
  を満たす場合のみエントリーするトレードシミュレーションを実施。
"""

import os
import sys
import time
import logging
from rich import print
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

# データ取得用モジュールのインポート
from data_fetch import fetch_multiple_bybit_data

# プロジェクトルートをパスに追加（必要に応じて調整）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# テクニカル指標計算用モジュールのインポート
from modules.prepocess.technical_indicators import technical_indicators

# TimeSeriesDataset（ウィンドウデータ作成用）のインポート
from modules.rnn_base.rnn_data_process import TimeSeriesDataset

# CatBoostのインポート
from catboost import CatBoostRegressor

# matplotlib のプロット設定
plt.style.use("default")
plt.rcParams["figure.figsize"] = (15, 12)

# ログ設定（INFOレベルのログを標準出力に出力）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------
# シミュレーション用設定パラメータ
# ---------------------------
config = {
    # CatBoostモデルのパス（マルチタスクモデル用）
    'catboost_model_path': 'models/catboost/catboost_model_final.cbm',
    # 取得対象のシンボル（例：BTCUSDT, ETHUSDT, XRPUSDT）
    'symbols': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT'],
    # 取得する足の間隔（15分足）
    'interval': '15',
    # APIで一度に取得するデータ数の上限
    'limit': 200,
    # 取得処理のループ回数
    'loops': 10,
    # シミュレーション開始時の初期資金
    'cash': 1000000,
    # 取引手数料
    'commission': 0.0002,
    # スリッページ
    'slippage': 0.0001,
    # CatBoost推論時に利用するウィンドウサイズ
    'window_size': 405,
    # CatBoost推論時のストライド
    'stride': 1,
}

#####################################
# CatBoost推論用関数（マルチタスク）
#####################################
def catboost_predict(df, model_path, window_size=405, stride=1):
    """
    CatBoostのマルチタスクモデルを用いて、入力データから6つの指標（
    LongReturn, ShortReturn, LongVolatility, ShortVolatility, LongMaxDD, ShortMaxDD）を予測する関数です。
    
    Args:
        df (DataFrame): 入力特徴量を含むデータフレーム
        model_path (str): 学習済みCatBoostモデルのパス
        window_size (int): 時系列ウィンドウサイズ（デフォルト: 405）
        stride (int): ウィンドウをずらすステップ幅（デフォルト: 1）
        
    Returns:
        df_pred (DataFrame): ウィンドウインデックスと6つの予測結果を持つデータフレーム。
            カラムは ['LongReturn', 'ShortReturn', 'LongVolatility', 'ShortVolatility', 'LongMaxDD', 'ShortMaxDD'] となります。
    """
    # TimeSeriesDatasetを用いてウィンドウデータを作成
    dataset = TimeSeriesDataset(predict_df=df, window_size=window_size, stride=stride)
    
    # 予測用特徴量は predict_features 属性に格納されている
    if dataset.predict_features is not None:
        feat_np = dataset.predict_features.numpy()
        N, W, F = feat_np.shape
        X_test = feat_np.reshape(N, W * F)
    else:
        raise ValueError("予測用特徴量が存在しません。")
    
    # CatBoostモデルの読み込み
    model = CatBoostRegressor()
    model.load_model(model_path)
    
    # 推論実施（出力は (N, 6) の配列を想定）
    predictions = model.predict(X_test)
    predictions = np.array(predictions)
    
    # 各ウィンドウの最終インデックスを取得
    ws = dataset.window_size
    st = dataset.stride
    window_indices = [df.index[i + ws - 1] for i in range(0, len(df) - ws, st)]
    
    if len(window_indices) != len(predictions):
        print(f"警告: ウィンドウ数 ({len(window_indices)}) と予測数 ({len(predictions)}) が一致しません。")
    
    # 予測結果をDataFrameに格納（6指標に対応）
    col_names = ['LongReturn', 'ShortReturn', 'LongVolatility', 'ShortVolatility', 'LongMaxDD', 'ShortMaxDD']
    df_pred = pd.DataFrame(predictions, index=window_indices, columns=col_names)
    
    return df_pred

############################################
# Backtrader 用データフィード定義（修正版）
############################################
class BybitCSV(bt.feeds.PandasData):
    """
    Backtraderで利用するカスタムPandasDataクラス。
    CatBoostの予測結果（6指標）も扱えるように拡張。
    """
    lines = ('LongReturn', 'ShortReturn', 'LongVolatility', 'ShortVolatility', 'LongMaxDD', 'ShortMaxDD')
    symbol = config['symbols'][0]
    params = (
        ('datetime', None),
        ('open', f'{symbol}_Open'),
        ('high', f'{symbol}_High'),
        ('low', f'{symbol}_Low'),
        ('close', f'{symbol}_Close'),
        ('volume', f'{symbol}_Volume'),
        ('LongReturn', 'LongReturn'),
        ('ShortReturn', 'ShortReturn'),
        ('LongVolatility', 'LongVolatility'),
        ('ShortVolatility', 'ShortVolatility'),
        ('LongMaxDD', 'LongMaxDD'),
        ('ShortMaxDD', 'ShortMaxDD'),
        ('openinterest', -1),
    )

#####################################################
# SMAフィルタ付き ロング・ショートエントリー戦略（教師ラベル生成用）
#####################################################
class SmaFilteredEntryExitLongShortStrategy(bt.Strategy):
    """
    SMAフィルタ付き ロング・ショート戦略:
      - SMAフィルタ: Close価格がSMAより上ならロング、下ならショートエントリー。
      - エントリールール: entry_intervalごとにエントリー。
      - さらに、エントリー時にはCatBoostの推論結果により、
            ロングの場合はLongReturnが0.001を超える、
            ショートの場合はShortReturnが -0.001未満であることを確認。
      - 保有期間: hold_periodバー後にクローズ（ストップロス・テイクプロフィット到達が先ならそちらを優先）
    
    トレード終了時に、各トレードの教師ラベルとして以下の6指標を算出:
      - ロングの場合: LongReturn, LongVolatility, LongMaxDD
      - ショートの場合: ShortReturn, ShortVolatility, ShortMaxDD
      ※ 該当しない指標は0.0とする。
    """
    params = (
        ('entry_interval', 1),       # 何バーおきにエントリーするか
        ('hold_period', 1),          # 保有期間（バー数）
        ('stop_loss', 0.02),         # ストップロス幅（例:2%）
        ('take_profit', 0.05),       # テイクプロフィット幅（例:5%）
        ('sma_period', 20),          # SMAの期間
        ('long_return_threshold', 0.00001),   # ロングエントリーの閾値
        ('short_return_threshold', -0.001), # ショートエントリーの閾値
    )

    def __init__(self):
        self.bar_count = 0
        self.open_trades = []   # 各トレードの情報を保持するリスト
        self.results = []       # 各トレード終了時に生成された教師ラベルのリスト

        # 移動平均インジケータ
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)

    def next(self):
        # 現在バーの日時とOHLC値の取得
        current_dt = self.data.datetime.datetime(0)
        o = self.data.open[0]
        h = self.data.high[0]
        l = self.data.low[0]
        c = self.data.close[0]

        ########################################
        # 1. オープン中トレードの管理
        ########################################
        trades_to_remove = []
        for trade in self.open_trades:
            # エントリー直後は1バー目は次バーから管理
            if trade.get('just_entered', False):
                trade['just_entered'] = False
                continue
            if trade['exited']:
                continue

            exit_price = None
            direction = trade['direction']

            if direction == 'long':
                if o >= trade['TP']:
                    exit_price = o
                elif o <= trade['SL']:
                    exit_price = o
                elif h >= trade['TP']:
                    exit_price = trade['TP']
                elif l <= trade['SL']:
                    exit_price = trade['SL']
            elif direction == 'short':
                if o <= trade['TP']:
                    exit_price = o
                elif o >= trade['SL']:
                    exit_price = o
                elif l <= trade['TP']:
                    exit_price = trade['TP']
                elif h >= trade['SL']:
                    exit_price = trade['SL']

            if exit_price is not None:
                trade['prices'].append(exit_price)
                trade['exited'] = True
                entry_price = trade['entry_price']
                ret = np.log(exit_price / entry_price) if direction == 'long' else np.log(entry_price / exit_price)
                prices_array = np.array(trade['prices'])

                # ボラティリティ計算（対数リターンの標準偏差）
                if len(prices_array) > 1:
                    log_returns = np.diff(np.log(prices_array))
                    vol = np.std(log_returns)
                else:
                    vol = 0.0

                # 最大ドローダウン計算（累積対数リターンに基づく）
                cum_log = np.log(prices_array / entry_price)
                running_max = np.maximum.accumulate(cum_log)
                drawdowns = running_max - cum_log
                max_dd = np.max(drawdowns)

                # 教師ラベルの生成
                if direction == 'long':
                    label = {
                        'entry_dt': trade['entry_dt'],
                        'LongReturn': ret,
                        'ShortReturn': 0.0,
                        'LongVolatility': vol,
                        'ShortVolatility': 0.0,
                        'LongMaxDD': max_dd,
                        'ShortMaxDD': 0.0,
                    }
                else:  # short
                    label = {
                        'entry_dt': trade['entry_dt'],
                        'LongReturn': 0.0,
                        'ShortReturn': ret,
                        'LongVolatility': 0.0,
                        'ShortVolatility': vol,
                        'LongMaxDD': 0.0,
                        'ShortMaxDD': max_dd,
                    }
                self.results.append(label)
                trades_to_remove.append(trade)
            else:
                # ストップ/テイクプロフィット未発動の場合、当バーのcloseを記録
                trade['prices'].append(c)
                trade['remaining'] -= 1
                if trade['remaining'] <= 0:
                    exit_price = c
                    trade['exited'] = True
                    entry_price = trade['entry_price']
                    ret = np.log(exit_price / entry_price) if direction == 'long' else np.log(entry_price / exit_price)
                    prices_array = np.array(trade['prices'])
                    if len(prices_array) > 1:
                        log_returns = np.diff(np.log(prices_array))
                        vol = np.std(log_returns)
                    else:
                        vol = 0.0
                    cum_log = np.log(prices_array / entry_price)
                    running_max = np.maximum.accumulate(cum_log)
                    drawdowns = running_max - cum_log
                    max_dd = np.max(drawdowns)

                    if direction == 'long':
                        label = {
                            'entry_dt': trade['entry_dt'],
                            'LongReturn': ret,
                            'ShortReturn': 0.0,
                            'LongVolatility': vol,
                            'ShortVolatility': 0.0,
                            'LongMaxDD': max_dd,
                            'ShortMaxDD': 0.0,
                        }
                    else:
                        label = {
                            'entry_dt': trade['entry_dt'],
                            'LongReturn': 0.0,
                            'ShortReturn': ret,
                            'LongVolatility': 0.0,
                            'ShortVolatility': vol,
                            'LongMaxDD': 0.0,
                            'ShortMaxDD': max_dd,
                        }
                    self.results.append(label)
                    trades_to_remove.append(trade)

        # クローズ済みトレードの削除
        self.open_trades = [t for t in self.open_trades if not t.get('exited', False)]

        ########################################
        # 2. バー数のカウント更新
        ########################################
        self.bar_count += 1

        ########################################
        # 3. 新規エントリー（entry_intervalごと）
        ########################################
        # エントリー時に、CatBoostの予測結果を利用して閾値チェックを実施
        # データフィードからはLongReturn, ShortReturnが参照可能とする
        if (self.bar_count - 1) % self.params.entry_interval == 0:
            if c > self.sma[0] and self.data.LongReturn[0] > self.p.long_return_threshold:
                entry_price = o
                trade = {
                    'entry_dt': current_dt,
                    'entry_price': entry_price,
                    'direction': 'long',
                    'remaining': self.p.hold_period,
                    'prices': [entry_price],
                    'SL': entry_price * (1 - self.p.stop_loss),
                    'TP': entry_price * (1 + self.p.take_profit),
                    'exited': False,
                    'just_entered': True,
                }
                self.open_trades.append(trade)
            elif c < self.sma[0] and self.data.ShortReturn[0] < self.p.short_return_threshold:
                entry_price = o
                trade = {
                    'entry_dt': current_dt,
                    'entry_price': entry_price,
                    'direction': 'short',
                    'remaining': self.p.hold_period,
                    'prices': [entry_price],
                    'SL': entry_price * (1 + self.p.stop_loss),
                    'TP': entry_price * (1 - self.p.take_profit),
                    'exited': False,
                    'just_entered': True,
                }
                self.open_trades.append(trade)

####################################
# メイン処理（シミュレーションの実行）
####################################
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addobserver(bt.observers.Broker, plot=False)

    df_combined = pd.DataFrame()
    symbols = config['symbols']

    # --- 各シンボルのデータ取得 ---
    for sym in symbols:
        logger.info(f"Fetching data for symbol: {sym}")
        df_sym = fetch_multiple_bybit_data(
            symbol=sym,
            interval=config['interval'],
            limit=config['limit'],
            loops=config['loops']
        )
        df_sym = df_sym.add_prefix(f"{sym}_")
        df_sym = technical_indicators(df_sym, sym)
        df_combined = pd.concat([df_combined, df_sym], axis=1)

    df_combined.dropna(inplace=True)
    df_combined.to_csv('forward_test/forwardtest.csv')

    # --- CatBoost 推論 ---
    df_catboost_predictions = catboost_predict(
        df_combined,
        config['catboost_model_path'],
        window_size=config['window_size'],
        stride=config['stride']
    )

    # CatBoostの予測結果を元のDataFrameに追加
    for col in df_catboost_predictions.columns:
        df_combined[col] = df_catboost_predictions[col]
    
    # 推論ウィンドウ分のデータが存在しない行を削除
    df_combined.dropna(inplace=True)
    
    # --- Backtrader用データフィード作成 ---
    forward_data = BybitCSV(dataname=df_combined, timeframe=bt.TimeFrame.Minutes, compression=15)
    cerebro.adddata(forward_data, name='ForwardTest')

    # --- 戦略の追加（SMAフィルタ付きロング・ショート戦略）
    cerebro.addstrategy(SmaFilteredEntryExitLongShortStrategy)
    cerebro.broker.setcash(config['cash'])
    cerebro.broker.setcommission(commission=config['commission'])
    cerebro.broker.set_slippage_fixed(fixed=config['slippage'])
    
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # --- パフォーマンス分析用アナライザーの追加 ---
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return', timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
    
    # --- シミュレーションの実行 ---
    results = cerebro.run()
    strat = results[0]

    # --- プロット処理 ---
    from plot import plot_results
    plot_results(cerebro, strat, config, logger)
