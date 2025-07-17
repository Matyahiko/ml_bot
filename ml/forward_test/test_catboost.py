#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# データ取得部分を切り出したファイルからインポート
from data_fetch import fetch_multiple_bybit_data

# 特徴量追加用の関数
from modules.preprocess.technical_indicators import technical_indicators
from modules.preprocess.add_lag_features import add_lag_features
from modules.preprocess.add_time_features import add_time_features 
from modules.preprocess.add_rolling_statistics import add_rolling_statistics

os.chdir("/app/ml_bot/ml")

config = {
    "symbols": ["BTCUSDT", "ETHUSDT", "XRPUSDT"],
    "interval": 15,
    "limit": 200,
    "loops": 50,
    "catboost_model_path": "models/catboost/catboost_regressor.cbm",
    # シミュレーション用パラメータ
    "cash": 100000,
    "commission": 0.001,
    "slippage": 0.001,
}

##################################
# 独自LinearRegressionインジケーター
##################################
class LinearRegression(bt.Indicator):
    lines = ('linreg',)
    params = (('period', 20),)
    
    def __init__(self):
        self.addminperiod(self.p.period)
    
    def next(self):
        period = self.p.period
        # 過去 period 個のデータ点を取得
        x = np.arange(period)
        y = np.array([self.data[i] for i in range(-period+1, 1)])
        slope, _ = np.polyfit(x, y, 1)
        self.lines.linreg[0] = slope

##################################
# カスタムデータフィード（必要最小限＋予測値追加）
##################################
class CustomPandasData(bt.feeds.PandasData):
    # max drawdown を除外し、log_return と volatility のみ利用
    lines = ('pred_log_return', 'pred_volatility',)
    params = (
        ('datetime', None),
        ('open', 'BTCUSDT_Open'),
        ('high', 'BTCUSDT_High'),
        ('low', 'BTCUSDT_Low'),
        ('close', 'BTCUSDT_Close'),
        ('volume', 'BTCUSDT_Volume'),
        ('pred_log_return', 'pred_log_return'),
        ('pred_volatility', 'pred_volatility'),
    )

##################################
# LazyBearSqueezeMomentum 指標
##################################
class LazyBearSqueezeMomentum(bt.Indicator):
    lines = ('squeeze', 'momentum',)
    params = (
        ('bb_period', 20),
        ('bb_devfactor', 2.0),
        ('kc_multiplier', 1.5),
    )
    plotinfo = dict(subplot=True)
    plotlines = dict(
        momentum=dict(color='blue'),
        squeeze=dict(marker='o', markersize=4, color='red'),
    )

    def __init__(self):
        # 中軸：SMA（BBとKCの共通基準）
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.bb_period)
        # 標準偏差
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.p.bb_period)
        # ボリンジャーバンド
        self.bb_upper = self.sma + self.p.bb_devfactor * self.std
        self.bb_lower = self.sma - self.p.bb_devfactor * self.std
        # ATR（KC計算用）
        self.atr = bt.indicators.ATR(self.data, period=self.p.bb_period)
        # ケルトナーチャネル
        self.kc_upper = self.sma + self.p.kc_multiplier * self.atr
        self.kc_lower = self.sma - self.p.kc_multiplier * self.atr
        # モメンタム： (Close - SMA) の線形回帰の傾きを利用
        self.linreg = LinearRegression(self.data.close - self.sma, period=self.p.bb_period)
        self.lines.momentum = self.linreg.lines.linreg

    def next(self):
        # BBがKC内に収まっている＝スクイーズ状態
        if self.bb_lower[0] > self.kc_lower[0] and self.bb_upper[0] < self.kc_upper[0]:
            self.lines.squeeze[0] = 1.0
        else:
            self.lines.squeeze[0] = 0.0

##################################
# LazyBearSqueezeMomentumStrategy（ロング・ショート両用、リスク調整後リターン条件追加）
##################################
class LazyBearSqueezeMomentumStrategy(bt.Strategy):
    params = (
        ('hold_period', 1),
        ('stop_loss', 0.02),
        ('take_profit', 0.05),
        ('risk_adj_threshold', 0.0001),
    )

    def __init__(self):
        self.order = None           # 発注中の注文を保持
        self.entry_bar = None       # エントリー時のバー番号を記録
        self.bar_count = 0          # 現在のバー番号
        self.smi = LazyBearSqueezeMomentum(self.data)

    def next(self):
        self.bar_count += 1
        current_dt = self.data.datetime.datetime(0)
        o = self.data.open[0]
        c = self.data.close[0]

        # すでにポジションを持っている場合、ホールド期間経過で決済（ポジションがあれば bracket 注文が自動でキャンセルされる）
        if self.position:
            if self.entry_bar is not None and (self.bar_count - self.entry_bar) >= self.p.hold_period:
                self.close()  # 決済注文を発注
            return  # ポジション中は新たな注文は出さない

        # すでに注文が出ている場合は何もしない
        if self.order:
            return

        # エントリーシグナルのチェック（前バーがスクイーズ中で、今バーでスクイーズが解除）
        if self.smi.squeeze[-1] == 1.0 and self.smi.squeeze[0] == 0.0:
            # ロングの場合（モメンタムが正）
            if self.smi.momentum[0] > 0:
                risk_adj_return = (self.data.pred_log_return[0] / self.data.pred_volatility[0]) if self.data.pred_volatility[0] != 0 else 0
                if risk_adj_return > self.p.risk_adj_threshold:
                    SL = o * (1 - self.p.stop_loss)
                    TP = o * (1 + self.p.take_profit)
                    # ロングの bracket 注文を発注（エントリー、ストップ、テイクプロフィット注文が同時に発注される）
                    self.order = self.buy_bracket(price=o, stopprice=SL, limitprice=TP)
            # ショートの場合（モメンタムが負）
            elif self.smi.momentum[0] < 0:
                risk_adj_return = (-self.data.pred_log_return[0] / self.data.pred_volatility[0]) if self.data.pred_volatility[0] != 0 else 0
                if risk_adj_return > self.p.risk_adj_threshold:
                    SL = o * (1 + self.p.stop_loss)
                    TP = o * (1 - self.p.take_profit)
                    # ショートの bracket 注文を発注
                    self.order = self.sell_bracket(price=o, stopprice=SL, limitprice=TP)

    def notify_order(self, order):
        # 注文完了時（約定、キャンセル、拒否）に呼ばれる
        if order.status in [order.Completed]:
            if order.exectype in [order.Market, order.Limit]:
                # エントリー注文が約定したら、現在のバー番号を記録
                if not self.position:
                    self.entry_bar = self.bar_count
        # 注文が完了、キャンセル、または拒否された場合、注文変数をクリア
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None
        
####################################
# メイン処理（シミュレーションの実行）
####################################
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addobserver(bt.observers.Broker, plot=False)

    df_combined = None  # 内部結合用に初期化
    symbols = config['symbols']

    # --- 各シンボルのデータ取得＆統合 ---
    for sym in symbols:
        df_sym = fetch_multiple_bybit_data(
            symbol=sym,
            interval=config['interval'],
            limit=config['limit'],
            loops=config['loops']
        )
        
        # タイムスタンプ処理
        if 'timestamp' in df_sym.columns:
            df_sym['timestamp'] = pd.to_datetime(df_sym['timestamp'])
            df_sym.set_index('timestamp', inplace=True)
            df_sym.sort_index(inplace=True)
        else:
            df_sym.index = pd.to_datetime(df_sym.index)
            df_sym.sort_index(inplace=True)
        
        # シンボルごとにカラム名にプレフィックスを追加
        df_sym = df_sym.add_prefix(f"{sym}_")
        
        # 内部結合（共通のタイムスタンプを基準）
        if df_combined is None:
            df_combined = df_sym
        else:
            df_combined = df_combined.join(df_sym, how='inner')

    # --- タイムスタンプの再設定 ---
    df_combined.reset_index(inplace=True)
    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'])
    df_combined.set_index('timestamp', inplace=True)
    df_combined.sort_index(inplace=True)

    # --- 特徴量追加 ---
    df_combined = add_time_features(df_combined)
    
    for sym in symbols:
        df_combined = technical_indicators(df_combined, sym)
        df_combined = add_lag_features(df_combined, sym, lags=[1, 2, 3])
        df_combined = add_rolling_statistics(df_combined, sym, windows=[5, 10, 20])
    
    df_combined.dropna(inplace=True)
    df_combined.to_csv('forward_test/forwardtest.csv')
    
    # --- CatBoost 推論 ---
    model = CatBoostRegressor(task_type='GPU', devices='0:1')
    model.load_model(config['catboost_model_path'])
    
    if hasattr(model, "feature_names_"):
        expected_features = model.feature_names_
        missing_features = set(expected_features) - set(df_combined.columns)
        if missing_features:
            raise ValueError(f"DataFrameに必要な特徴量が不足しています: {missing_features}")
        df_combined = df_combined[expected_features]
    else:
        pass  # model.feature_names_ が取得できなかった場合、事前に特徴量リストを確認してください。
    
    predictions = model.predict(df_combined)
    print(predictions)
    df_combined['pred_log_return'] = predictions[:, 0]
    df_combined['pred_volatility'] = predictions[:, 1]
    
    # --- Backtrader用データフィード作成 ---
    forward_data = CustomPandasData(dataname=df_combined, timeframe=bt.TimeFrame.Minutes, compression=15)
    cerebro.adddata(forward_data, name='ForwardTest')

    # --- 戦略の追加（リスク調整後リターン条件付きロング・ショート戦略） ---
    cerebro.addstrategy(LazyBearSqueezeMomentumStrategy)
    cerebro.broker.setcash(config['cash'])
    cerebro.broker.setcommission(commission=config['commission'])
    cerebro.broker.set_slippage_fixed(fixed=config['slippage'])
    
    # パフォーマンス分析用アナライザーの追加
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
    plot_results(cerebro, strat, config, None)
