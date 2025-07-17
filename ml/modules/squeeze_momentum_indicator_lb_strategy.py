#for backtest
#squeeze_momentum_indicator_lb_strategy.py

import backtrader as bt
import numpy as np

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
        ('bb_period', 6),              # 収束検出期間を短縮して頻度増
        ('kc_multiplier', 0.9),        # KCをさらに狭めて頻度を高める
        ('bb_multiplier', 1.2),        # ボリンジャーバンド幅を明示的に調整
    )

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.bb_period)
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.p.bb_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.bb_period)

        self.bb_upper = self.sma + self.p.bb_multiplier * self.std
        self.bb_lower = self.sma - self.p.bb_multiplier * self.std

        self.kc_upper = self.sma + self.p.kc_multiplier * self.atr
        self.kc_lower = self.sma - self.p.kc_multiplier * self.atr

        self.linreg = LinearRegression(self.data.close - self.sma, period=8)  # 短期に変更
        
        super().__init__()

    def next(self):
        if self.bb_lower[0] > self.kc_lower[0] and self.bb_upper[0] < self.kc_upper[0]:
            self.lines.squeeze[0] = 1.0
        else:
            self.lines.squeeze[0] = 0.0

        self.lines.momentum[0] = self.linreg[0]


class LazyBearSqueezeMomentumStrategy(bt.Strategy):
    params = (
        ('stop_loss', 0.025),
        ('take_profit', 0.03),
        ('time_exit', 3),          # 保持期間（本数）
        ('threshold', 0.012),      # モメンタム閾値
    )

    def __init__(self):
        self.smi = LazyBearSqueezeMomentum(self.data)
        self.order_parent = None
        self.entry_bar = None

    def next(self):
        entry_price = self.data.close[0]

        # ポジション保有中
        if self.position:
            if self.entry_bar is not None and (len(self) - self.entry_bar >= self.p.time_exit):
                self.close()
            return  # ポジション中は新規エントリーなし

        if self.order_parent:
            return  # 注文が未約定なら何もしない

        squeeze_released = (self.smi.squeeze[-1] == 1.0 and self.smi.squeeze[0] == 0.0)
        squeeze_active_strong_momentum = (self.smi.squeeze[0] == 1.0 and abs(self.smi.momentum[0]) > self.p.threshold)

        entry_price = self.data.close[0]
        orders = None

        if squeeze_released or squeeze_active_strong_momentum:
            if self.smi.momentum[0] > 0:
                SL = entry_price * (1 - self.p.stop_loss)
                TP = entry_price * (1 + self.p.take_profit)
                orders = self.buy_bracket(price=entry_price, stopprice=SL, limitprice=TP)
            elif self.smi.momentum[0] < 0:
                SL = entry_price * (1 + self.p.stop_loss)
                TP = entry_price * (1 - self.p.take_profit)
                orders = self.sell_bracket(price=entry_price, stopprice=SL, limitprice=TP)
            else:
                orders = None

            if orders:
                self.order_parent = orders[0]
                self.order_stop   = orders[1]
                self.order_limit  = orders[2]

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy() or order.issell():
                # エントリーが完了したタイミングでセットするのが正しい
                if not self.position:  # 新規エントリー約定時のみentry_barをセット
                    self.entry_bar = len(self)

        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order_parent = None
            self.order_stop = None
            self.order_limit = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pass  # entry_barはここではリセットしない！

