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
# 修正版RSIとEMAを組み合わせた戦略
##################################
class RSIMA_Strategy(bt.Strategy):
    params = (
        ('rsi_period', 10),                  # RSI期間を少し短縮
        ('ma_period', 12),                   # EMA期間を少し短縮
        ('rsi_long_threshold', 60),          # RSIロング閾値を少し緩める
        ('rsi_short_threshold', 40),         # RSIショート閾値を少し緩める
        ('stop_loss', 0.03),                 # SLは変更なし
        ('take_profit', 0.05),               # TPも変更なし
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.ma = bt.indicators.EMA(self.data.close, period=self.p.ma_period)
        self.crossover = bt.indicators.CrossOver(self.data.close, self.ma)

        self.order_parent = None
        self.order_stop = None
        self.order_limit = None

    def next(self):
        if self.position:
            return  # ポジションがあればSL/TP決済に任せる

        if self.order_parent:
            return  # 未決済の注文があれば新規注文しない

        entry_price = self.data.close[0]  # 終値ベースでエントリー

        if self.crossover[0] > 0 and self.rsi[0] > self.p.rsi_long_threshold:
            SL = entry_price * (1 - self.p.stop_loss)
            TP = entry_price * (1 + self.p.take_profit)
            orders = self.buy_bracket(price=entry_price, stopprice=SL, limitprice=TP)
            if orders:
                self.order_parent, self.order_stop, self.order_limit = orders

        elif self.crossover[0] < 0 and self.rsi[0] < self.p.rsi_short_threshold:
            SL = entry_price * (1 + self.p.stop_loss)
            TP = entry_price * (1 - self.p.take_profit)
            orders = self.sell_bracket(price=entry_price, stopprice=SL, limitprice=TP)
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
