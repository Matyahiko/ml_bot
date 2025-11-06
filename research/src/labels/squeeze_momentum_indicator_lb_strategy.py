# squeeze_momentum_strategy.py

import backtrader as bt
import numpy as np

class LinearRegression(bt.Indicator):
    lines = ('linreg',)
    params = (('period', 20),)

    def __init__(self) -> None:
        self.addminperiod(self.p.period)

    def next(self) -> None:
        y = np.array([self.data[i] for i in range(-self.p.period+1, 1)])
        slope, _ = np.polyfit(np.arange(self.p.period), y, 1)
        self.lines.linreg[0] = slope

class CustomPandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),
        ('open', 'BTCUSDT_Open'),
        ('high', 'BTCUSUT_High'),
        ('low', 'BTCUSDT_Low'),
        ('close', 'BTCUSDT_Close'),
        ('volume', 'BTCUSDT_Volume'),
    )

class LazyBearSqueezeMomentum(bt.Indicator):
    lines = ('squeeze', 'momentum')
    params = (('bb_period', 6), ('kc_multiplier', 0.9), ('bb_multiplier', 1.2))

    def __init__(self) -> None:
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.bb_period)
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.p.bb_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.bb_period)
        self.bb_upper = self.sma + self.p.bb_multiplier * self.std
        self.bb_lower = self.sma - self.p.bb_multiplier * self.std
        self.kc_upper = self.sma + self.p.kc_multiplier * self.atr
        self.kc_lower = self.sma - self.p.kc_multiplier * self.atr
        self.linreg = LinearRegression(self.data.close - self.sma, period=8)

    def next(self) -> None:
        if self.bb_lower[0] > self.kc_lower[0] and self.bb_upper[0] < self.kc_upper[0]:
            self.lines.squeeze[0] = 1.0
        else:
            self.lines.squeeze[0] = 0.0
        self.lines.momentum[0] = self.linreg[0]

class LazyBearSqueezeMomentumStrategy(bt.Strategy):
    params = (('stop_loss', 0.025), ('take_profit', 0.03), ('time_exit', 3), ('threshold', 0.012))

    def __init__(self) -> None:
        self.smi = LazyBearSqueezeMomentum(self.data)
        self.order_parent = None
        self.entry_bar = None

    def next(self) -> None:
        if self.position:
            if self.entry_bar is not None and len(self) - self.entry_bar >= self.p.time_exit:
                self.close()
            return
        if self.order_parent:
            return
        released = self.smi.squeeze[-1] == 1.0 and self.smi.squeeze[0] == 0.0
        strong = self.smi.squeeze[0] == 1.0 and abs(self.smi.momentum[0]) > self.p.threshold
        if released or strong:
            price = self.data.close[0]
            if self.smi.momentum[0] > 0:
                sl = price * (1 - self.p.stop_loss); tp = price * (1 + self.p.take_profit)
                orders = self.buy_bracket(price=price, stopprice=sl, limitprice=tp)
            else:
                sl = price * (1 + self.p.stop_loss); tp = price * (1 - self.p.take_profit)
                orders = self.sell_bracket(price=price, stopprice=sl, limitprice=tp)
            if orders:
                self.order_parent, self.order_stop, self.order_limit = orders

    def notify_order(self, order: bt.Order) -> None:
        if order.status == order.Completed and not self.position:
            self.entry_bar = len(self)
        if order.status in (order.Completed, order.Canceled, order.Rejected):
            self.order_parent = self.order_stop = self.order_limit = None

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isclosed:
            pass
