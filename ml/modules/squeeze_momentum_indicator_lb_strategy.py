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
        ('hold_period', 5),
        ('stop_loss', 0.02),
        ('take_profit', 0.05),
    )

    def __init__(self):
        # 初期化: 取引状態のフラグとタイプを設定
        self.trade_flag = False
        self.trade_type = None

        self.bar_count = 0
        self.open_trades = []  # オープン中のトレードを記録
        self.raw_trades = []   # 終了したトレードの記録
        self.smi = LazyBearSqueezeMomentum(self.data)  # 指標の初期化

    def next(self):
        current_dt = self.data.datetime.datetime(0)
        o = self.data.open[0]
        c = self.data.close[0]

        # 既存のオープンポジションのチェックと更新
        for trade in self.open_trades.copy():
            if trade.get('just_entered', False):
                trade['just_entered'] = False
                continue

            trade['remaining'] -= 1
            if trade['remaining'] <= 0 or self.should_exit_trade(trade, c):
                trade['exit_price'] = c
                trade['exit_dt'] = current_dt
                trade['exited'] = True
                self.raw_trades.append(trade)
                self.open_trades.remove(trade)

        # ポジションがなくなったらフラグをリセット
        if not self.open_trades:
            self.trade_flag = False
            self.trade_type = None

        self.bar_count += 1

        # エントリーシグナル：前バーがスクイーズ中で、今バーでスクイーズ解除されたタイミング
        # ※複数ポジションを持たないため、すでにポジションがある場合は新規エントリーしない
        if not self.trade_flag and self.bar_count > 1:
            if self.smi.squeeze[-1] == 1.0 and self.smi.squeeze[0] == 0.0:
                if self.smi.momentum[0] > 0:
                    self._enter_trade(current_dt, o, 'long')
                elif self.smi.momentum[0] < 0:
                    self._enter_trade(current_dt, o, 'short')

    def _enter_trade(self, current_dt, entry_price, direction):
        if direction == 'long':
            SL = entry_price * (1 - self.p.stop_loss)
            TP = entry_price * (1 + self.p.take_profit)
        else:  # 'short'
            SL = entry_price * (1 + self.p.stop_loss)
            TP = entry_price * (1 - self.p.take_profit)
        trade = {
            'entry_dt': current_dt,
            'entry_price': entry_price,
            'direction': direction,
            'remaining': self.p.hold_period,
            'exit_price': None,
            'SL': SL,
            'TP': TP,
            'exited': False,
            'just_entered': True,
        }
        self.open_trades.append(trade)
        
        # エントリー時にフラグを更新（既にポジションがあるため新規エントリーは防ぐ）
        self.trade_flag = True
        self.trade_type = direction

    def should_exit_trade(self, trade, price):
        if trade['direction'] == 'long':
            return price >= trade['TP'] or price <= trade['SL']
        else:  # 'short'
            return price <= trade['TP'] or price >= trade['SL']