import backtrader as bt
import numpy as np

##################################
# 独自LinearRegressionインジケーター（変更なし）
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
# 差分を計算するカスタムインジケーター
##################################
class DifferenceIndicator(bt.Indicator):
    lines = ('diff',)
    
    def __init__(self, other):
        # other は引く側のライン（ここでは SMA）
        self.other = other
        self.addminperiod(1)
    
    def next(self):
        self.lines.diff[0] = self.data[0] - self.other[0]

##################################
# カスタムデータフィード（変更なし）
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
# LazyBearSqueezeMomentum 指標（修正版）
##################################
class LazyBearSqueezeMomentum(bt.Indicator):
    lines = ('squeeze', 'momentum',)
    params = (
        ('bb_period', 20),
        ('bb_devfactor', 2.0),
        ('kc_multiplier', 1.5),
        ('hold_period', None),   # 戦略から渡されるパラメータを吸収
        ('stop_loss', None),     # 同上
        ('take_profit', None),   # 同上
    )
    plotinfo = dict(subplot=True)
    plotlines = dict(
        momentum=dict(color='blue'),
        squeeze=dict(marker='o', markersize=4, color='red'),
    )

    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.bb_period)
        self.std = bt.indicators.StandardDeviation(self.data.close, period=self.p.bb_period)
        self.bb_upper = self.sma + self.p.bb_devfactor * self.std
        self.bb_lower = self.sma - self.p.bb_devfactor * self.std
        self.atr = bt.indicators.ATR(self.data, period=self.p.bb_period)
        self.kc_upper = self.sma + self.p.kc_multiplier * self.atr
        self.kc_lower = self.sma - self.p.kc_multiplier * self.atr
        
        # ここで、直接算術演算せずに DifferenceIndicator を使う
        self.diff = DifferenceIndicator(self.sma)
        self.linreg = LinearRegression(self.diff, period=self.p.bb_period)
        # 直接 self.lines.momentum へ代入はせず、 next() 内で更新する

    def next(self):
        # linreg の現在の値を momentum ラインに設定
        self.lines.momentum[0] = self.linreg[0]
        # シンプルなルール：終値がSMAより上なら squeeze を 1、下なら 0 にする
        if self.data.close[0] > self.sma[0]:
            self.lines.squeeze[0] = 1.0
        else:
            self.lines.squeeze[0] = 0.0
            
##################################
# シンプルな戦略：LazyBearSqueezeMomentumStrategy（エントリー条件をモメンタムの符号のみに変更）
##################################
class LazyBearSqueezeMomentumStrategy(bt.Strategy):
    params = (
        ('hold_period', 5),     # 保有期間（例：5バー）
        ('stop_loss', 0.02),
        ('take_profit', 0.05),
    )

    def __init__(self):
        self.smi = LazyBearSqueezeMomentum(self.data)
        self.bar_count = 0       # 全体のバー数カウント
        self.entry_bar = None    # エントリー時のバー番号

        self.order_parent = None
        self.order_stop = None
        self.order_limit = None

    def next(self):
        self.bar_count += 1

        # ポジションを持っている場合、保有期間に達したら決済
        if self.position:
            if self.entry_bar is not None and (self.bar_count - self.entry_bar) >= self.p.hold_period:
                if self.order_stop and self.order_stop.status in [bt.Order.Submitted, bt.Order.Accepted]:
                    self.cancel(self.order_stop)
                if self.order_limit and self.order_limit.status in [bt.Order.Submitted, bt.Order.Accepted]:
                    self.cancel(self.order_limit)
                self.close()  # マーケットで決済

        # ポジションがなく、エントリー注文が未発注の場合、モメンタムのみでエントリー判定
        if not self.position and not self.order_parent:
            entry_price = self.data.open[0]
            if self.smi.momentum[0] > 0:
                # ロングエントリー
                SL = entry_price * (1 - self.p.stop_loss)
                TP = entry_price * (1 + self.p.take_profit)
                orders = self.buy_bracket(
                    price=entry_price,
                    stopprice=SL,
                    limitprice=TP
                )
                if orders:
                    self.order_parent = orders[0]
                    self.order_stop   = orders[1]
                    self.order_limit  = orders[2]
            elif self.smi.momentum[0] < 0:
                # ショートエントリー
                SL = entry_price * (1 + self.p.stop_loss)
                TP = entry_price * (1 - self.p.take_profit)
                orders = self.sell_bracket(
                    price=entry_price,
                    stopprice=SL,
                    limitprice=TP
                )
                if orders:
                    self.order_parent = orders[0]
                    self.order_stop   = orders[1]
                    self.order_limit  = orders[2]

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order == self.order_parent:
                self.entry_bar = self.bar_count
                self.order_parent = None
        elif order.status in [order.Canceled, order.Rejected]:
            if order == self.order_parent:
                self.order_parent = None
            if order == self.order_stop:
                self.order_stop = None
            if order == self.order_limit:
                self.order_limit = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.entry_bar = None
            self.order_stop = None
            self.order_limit = None
