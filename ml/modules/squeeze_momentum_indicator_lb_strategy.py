import backtrader as bt
import numpy as np
import pandas as pd

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
# カスタムデータフィード（必要最小限）
##################################
class CustomPandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'BTCUSDT_Open'),
        ('high', 'BTCUSDT_High'),
        ('low', 'BTCUSDT_Low'),
        ('close', 'BTCUSDT_Close'),
        ('volume', 'BTCUSDT_Volume'),
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
# LazyBearSqueezeMomentumStrategy（ロング・ショート両用）
##################################
class LazyBearSqueezeMomentumStrategy(bt.Strategy):
    params = (
        ('hold_period', 5),   # 保有期間（バー数）
        ('stop_loss', 0.02),  # ストップロスの割合
        ('take_profit', 0.05),# テイクプロフィットの割合
    )
    
    def log(self, txt, dt=None):
        """ログ出力用のユーティリティ"""
        dt = dt or self.data.datetime.datetime(0)
        print(f'{dt.isoformat()} - {txt}')
    
    def __init__(self):
        self.smi = LazyBearSqueezeMomentum(self.data)
        self.order = None  # 発注中の注文を保持
        self.entry_bar = None  # エントリーが約定したバー番号を記録

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # まだ処理中

        # 注文が完了した場合
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}')
            # エントリー注文完了時にバー番号を記録（後の保有期間判定に利用）
            if not self.position:
                # クローズ注文の場合は無視
                pass
            else:
                if self.entry_bar is None:
                    self.entry_bar = len(self)
            self.order = None

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'OPERATION PROFIT, GROSS {trade.pnl:.2f}, NET {trade.pnlcomm:.2f}')
            self.entry_bar = None  # トレード終了時にエントリーバーをリセット

    def next(self):
        # すでに注文中の場合は何もしない
        if self.order:
            return

        # 未ポジション時にエントリーシグナルの判定
        if not self.position:
            # エントリーシグナル：前バーがスクイーズ状態、今バーでスクイーズ解除
            if self.smi.squeeze[-1] == 1.0 and self.smi.squeeze[0] == 0.0:
                entry_price = self.data.open[0]
                if self.smi.momentum[0] > 0:
                    # ロングエントリー：ストップロス／テイクプロフィットを bracket 注文で設定
                    sl = entry_price * (1 - self.p.stop_loss)
                    tp = entry_price * (1 + self.p.take_profit)
                    self.log(f'Long Entry Signal at Price: {entry_price:.2f}')
                    self.order = self.buy_bracket(
                        size=1,
                        stopprice=sl,
                        limitprice=tp
                    )
                elif self.smi.momentum[0] < 0:
                    # ショートエントリー
                    sl = entry_price * (1 + self.p.stop_loss)
                    tp = entry_price * (1 - self.p.take_profit)
                    self.log(f'Short Entry Signal at Price: {entry_price:.2f}')
                    self.order = self.sell_bracket(
                        size=1,
                        stopprice=sl,
                        limitprice=tp
                    )
        else:
            # ポジション保有中の場合、保有期間が経過していたらクローズ注文を発注
            if self.entry_bar is not None and (len(self) - self.entry_bar) >= self.p.hold_period:
                self.log(f'Hold period reached, closing position at {self.data.open[0]:.2f}')
                self.order = self.close()