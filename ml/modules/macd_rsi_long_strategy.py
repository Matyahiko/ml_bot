import backtrader as bt

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
# MACDブルクロスオーバーとRSIオーバーソールド（5本前）に基づくロング戦略
##################################
class MACD_RSI_Long_Strategy(bt.Strategy):
    params = (
        ('rsi_period', 14),
        ('rsi_oversold', 30),
        ('macd_fast', 12),
        ('macd_slow', 26),
        ('macd_signal', 9),
        ('stop_loss', 0.03),
        ('take_profit', 0.05),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.macd_fast,
            period_me2=self.p.macd_slow,
            period_signal=self.p.macd_signal
        )
        # MACDとシグナルラインのクロスを明確にする
        self.macd_cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        self.order_parent = None
        self.order_stop = None
        self.order_limit = None

    def next(self):
        if len(self) < 5:
            return

        if self.position or self.order_parent:
            return

        entry_price = self.data.close[0]

        recent_rsi = [self.rsi[-i] for i in range(5)]
        rsi_condition = any(rsi < self.p.rsi_oversold for rsi in recent_rsi)

        # MACDがシグナルラインを上抜ける瞬間を条件に戻す（クロスを利用）
        if self.macd_cross[0] > 0 and rsi_condition:
            SL = entry_price * (1 - self.p.stop_loss)
            TP = entry_price * (1 + self.p.take_profit)
            orders = self.buy_bracket(price=entry_price, stopprice=SL, limitprice=TP)
            if orders:
                self.order_parent, self.order_stop, self.order_limit = orders

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            if order == self.order_parent:
                self.order_parent = None
            if order == self.order_stop:
                self.order_stop = None
            if order == self.order_limit:
                self.order_limit = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.order_parent = None
            self.order_stop = None
            self.order_limit = None

