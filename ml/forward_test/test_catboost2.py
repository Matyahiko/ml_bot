
#for forward test
#squeeze_momentum_indicator_lb_strategy.py
import os
import sys
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
from dataclasses import dataclass
from catboost import CatBoostRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# データ取得部分を切り出したファイルからインポート
from data_fetch import fetch_multiple_bybit_data

# 特徴量追加用の関数
from modules.preprocess.technical_indicators import technical_indicators
from modules.preprocess.add_lag_features import add_lag_features
from modules.preprocess.add_time_features import add_time_features 
from modules.preprocess.add_rolling_statistics import add_rolling_statistics

# 正規化用の関数をインポート
from modules.preprocess.normalization import fit_transform_scaler, transform_scaler


os.chdir("/app/ml_bot/ml")

config = {
    "symbols": ["BTCUSDT", "ETHUSDT", "XRPUSDT"],
    "interval": 15,
    "limit": 200,
    "loops": 50,
    "catboost_model_path": "models/catboost/catboost_regressor.cbm",
    # シミュレーション用パラメータ
    "cash": 100000000,
    "commission": 0.001,
    "slippage": 0.001,
}

##################################
# カスタムデータフィード（ML予測値を含む）
##################################
class CustomPandasData(bt.feeds.PandasData):
    # 外部で計算済みのadj_returnもラインとして定義
    lines = ('pred_log_return', 'pred_volatility', 'adj_return',)
    params = (
        ('datetime', None),
        ('open', 'BTCUSDT_Open'),
        ('high', 'BTCUSDT_High'),
        ('low', 'BTCUSDT_Low'),
        ('close', 'BTCUSDT_Close'),
        ('volume', 'BTCUSDT_Volume'),
        ('pred_log_return', 'pred_log_return'),
        ('pred_volatility', 'pred_volatility'),
        ('adj_return', 'adj_return'),
    )

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
        x = np.arange(period)
        y = np.array([self.data[i] for i in range(-period+1, 1)])
        slope, _ = np.polyfit(x, y, 1)
        self.lines.linreg[0] = slope

##################################
# LazyBearSqueezeMomentum 指標
##################################
class LazyBearSqueezeMomentum(bt.Indicator):
    lines = ('squeeze', 'momentum',)
    params = (
        ('bb_period', 6),        # 収束検出期間（短期）
        ('kc_multiplier', 0.9),  # KC幅を狭める
        ('bb_multiplier', 1.2),  # ボリンジャーバンド幅の調整
    )

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.bb_period)
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.p.bb_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.bb_period)

        self.bb_upper = self.sma + self.p.bb_multiplier * self.std
        self.bb_lower = self.sma - self.p.bb_multiplier * self.std

        self.kc_upper = self.sma + self.p.kc_multiplier * self.atr
        self.kc_lower = self.sma - self.p.kc_multiplier * self.atr

        # 短期の線形回帰（価格-移動平均線）の傾き
        self.linreg = LinearRegression(self.data.close - self.sma, period=8)
        super().__init__()

    def next(self):
        # スクイーズ状態：ボリンジャーバンドがKC内に収まっている場合
        if self.bb_lower[0] > self.kc_lower[0] and self.bb_upper[0] < self.kc_upper[0]:
            self.lines.squeeze[0] = 1.0
        else:
            self.lines.squeeze[0] = 0.0
        # モメンタムは線形回帰の傾き
        self.lines.momentum[0] = self.linreg[0]

##################################
# LazyBearSqueezeMomentum戦略（MLによるポジションサイズ決定＋タイムエグジット付き）
##################################

@dataclass
class BracketOrder:
    entry: any = None
    stop: any = None
    limit: any = None
    submitted_bar: int = None
    entry_bar: int = None

class LazyBearSqueezeMomentumStrategy(bt.Strategy):
    params = (
        ('time_exit', 3),      # ポジション保有時間（バー数）),
        ('threshold', 5.0),     # リスク調整後リターンの閾値
        ('base_risk', 0.3),     # 資金のn%をリスク許容額とする
        ('sl_coef', 2.0),       # SL係数（予測ボラティリティに乗じる）
        ('tp_coef', 4.0),       # TP係数（予測ボラティリティに乗じる）
        ('order_timeout', 2),   # 注文タイムアウト：保留中の注文が3バー以上経過した場合キャンセルする
    )

    def __init__(self):
        # インジケータ
        self.smi = LazyBearSqueezeMomentum(self.data)
        # ブラケット注文管理用のオブジェクト
        self.bracket_order = BracketOrder()
        # デバッグ・記録用
        self.step_records = []
        self.decision_records = []

    def check_entry_signal(self):
        """エントリーシグナルの判定"""
        squeeze_released = (self.smi.squeeze[-1] == 1.0 and self.smi.squeeze[0] == 0.0)
        squeeze_active_strong_momentum = (self.smi.squeeze[0] == 1.0 and abs(self.smi.momentum[0]) > self.p.threshold)
        if not (squeeze_released or squeeze_active_strong_momentum):
            return None

        risk_signal = self.data.adj_return[0]
        if risk_signal > self.p.threshold:
            return 'long'
        elif risk_signal < -self.p.threshold:
            return 'short'
        return None

    def submit_bracket_order(self, trade_dir, risk_signal):
        """ブラケット注文の発注処理"""
        entry_price = self.data.close[0]
        cash = self.broker.getcash()
        position_size = (cash * self.p.base_risk) / (abs(risk_signal) * entry_price)
        sl_distance = abs(risk_signal) * self.p.sl_coef
        tp_distance = abs(risk_signal) * self.p.tp_coef

        if trade_dir == 'long':
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
            orders = self.buy_bracket(
                size=position_size,
                price=entry_price,
                stopprice=sl_price,
                limitprice=tp_price
            )
        else:  # 'short'
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
            orders = self.sell_bracket(
                size=position_size,
                price=entry_price,
                stopprice=sl_price,
                limitprice=tp_price
            )

        if orders:
            self.bracket_order = BracketOrder(
                entry=orders[0],
                stop=orders[1],
                limit=orders[2],
                submitted_bar=len(self),
                entry_bar=len(self)
            )

    def manage_order_timeout(self):
        """注文タイムアウトのチェック処理"""
        bo = self.bracket_order
        if bo.entry and bo.submitted_bar is not None:
            if len(self) - bo.submitted_bar >= self.p.order_timeout:
                self.cancel(bo.entry)
                print(f"[{self.data.datetime.datetime()}] Order timed out and cancelled.")
                if bo.stop:
                    self.cancel(bo.stop)
                if bo.limit:
                    self.cancel(bo.limit)
                self.bracket_order = BracketOrder()

    def manage_position(self):
        """ポジション保有中のタイムエグジット処理"""
        bo = self.bracket_order
        if self.position and bo.entry_bar is not None:
            if len(self) - bo.entry_bar >= self.p.time_exit:
                print(f"[{self.data.datetime.datetime()}] Time exit. Closing position.")
                self.close()
                if bo.stop:
                    self.cancel(bo.stop)
                if bo.limit:
                    self.cancel(bo.limit)
                self.bracket_order = BracketOrder()

    def record_trade_decision(self, trade_dir, risk_signal):
        """トレード判断の記録"""
        self.decision_records.append({
            "datetime": self.data.datetime.datetime(),
            "close_price": self.data.close[0],
            "pred_log_return": self.data.pred_log_return[0],
            "pred_volatility": self.data.pred_volatility[0],
            "adj_return": self.data.adj_return[0],
            "risk_signal": risk_signal,
            "trade_direction": trade_dir,
            "momentum": self.smi.momentum[0],
            "squeeze": self.smi.squeeze[0],
        })

    def next(self):
        # (1) 注文タイムアウト処理
        self.manage_order_timeout()

        # (2) ポジション管理（タイムエグジット）
        self.manage_position()

        # ポジション保有中または注文中の場合は新規エントリーしない
        if self.position or self.bracket_order.entry:
            return

        # (3) エントリーシグナルのチェック
        trade_dir = self.check_entry_signal()
        if not trade_dir:
            return

        risk_signal = self.data.adj_return[0]

        # (4) トレード判断の記録
        self.record_trade_decision(trade_dir, risk_signal)

        # (5) ブラケット注文発注
        self.submit_bracket_order(trade_dir, risk_signal)

    def notify_order(self, order):
        print(f"[notify_order] {self.data.datetime.datetime()} Status={order.getstatusname()}, Ref={order.ref}, "
              f"isbuy={order.isbuy()}, Size={order.created.size}")
        bo = self.bracket_order

        if order.status in [order.Completed]:
            if order is bo.entry:
                print(" -> Entry order completed.")
                bo.entry = None
                bo.submitted_bar = None
                # エントリー約定後は entry_bar はそのまま保持
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if order is bo.entry:
                print(" -> Entry order cancelled/rejected.")
                self.bracket_order = BracketOrder()  # 全リセット
            if order is bo.stop:
                print(" -> Stop order cancelled/rejected.")
                bo.stop = None
            if order is bo.limit:
                print(" -> Limit order cancelled/rejected.")
                bo.limit = None

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f"[notify_trade] {self.data.datetime.datetime()} Trade closed, PnL: {trade.pnl}")
            self.bracket_order = BracketOrder()

    def stop(self):
        # バックテスト終了時に呼ばれるメソッド
        step_df = pd.DataFrame(self.step_records)
        step_df.to_csv("forward_test/step_records.csv", index=False)
        print("Step Records:")
        print(step_df.head(10))

        decision_df = pd.DataFrame(self.decision_records)
        decision_df.to_csv("forward_test/forwardtest_decision_predictions.csv", index=False)
        print("Decision Records:")
        print(decision_df.head(10))



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
        df_combined = add_rolling_statistics(df_combined, sym, windows=[5, 10, 20])
        df_combined = add_lag_features(df_combined, sym, lags=[1, 2, 3])
    
    # --- 正規化 ---
    import joblib
    # 現在の正規化コードを置き換える
    scaler_obj = joblib.load('storage/kline/minmax_scaler.joblib')
    df_combined = transform_scaler(df_combined, scaler_obj)
    
    df_combined.dropna(inplace=True)
    
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
    df_combined['pred_log_return'] = predictions[:, 0]
    df_combined['pred_volatility'] = predictions[:, 1]
    epsilon = 1e-8  # ゼロ除算防止用
    df_combined['adj_return'] = df_combined['pred_log_return'] / (df_combined['pred_volatility'] + epsilon)
    
    print(df_combined[['pred_log_return', 'pred_volatility', 'adj_return']].head())
    df_combined[['pred_log_return', 'pred_volatility', 'adj_return']].to_csv("forward_test/forwardtest_prediction.csv")
    
    # --- Backtrader用データフィード作成 ---
    forward_data = CustomPandasData(dataname=df_combined, timeframe=bt.TimeFrame.Minutes, compression=15)
    cerebro.adddata(forward_data, name='ForwardTest')

    # --- 戦略の追加 ---
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
