import backtrader as bt

##################################
# カスタムデータフィード（必要最小限）
##################################
class CustomPandasData(bt.feeds.PandasData):
    params = (
        ('datetime', None),  # DataFrameのインデックスを日時として使用
        ('open', 'BTCUSDT_Open'),
        ('high', 'BTCUSDT_High'),
        ('low', 'BTCUSDT_Low'),
        ('close', 'BTCUSDT_Close'),
        ('volume', 'BTCUSDT_Volume'),
    )

##################################
# SmaFilteredStrategy（ロング・ショート両用）
##################################
class SmaFilteredStrategy(bt.Strategy):
    params = (
        ('entry_interval', 1),    # エントリーするバー間隔
        ('hold_period', 1),       # 保有期間（バー数）
        ('stop_loss', 0.02),      # ストップロス幅
        ('take_profit', 0.05),    # テイクプロフィット幅
        ('sma_period', 20),       # SMAの期間
    )

    def __init__(self):
        self.bar_count = 0
        self.open_trades = []   # オープン中のトレード記録
        self.raw_trades = []    # 終了したトレードの生データを保持
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.sma_period)

    def next(self):
        current_dt = self.data.datetime.datetime(0)
        o = self.data.open[0]
        h = self.data.high[0]
        l = self.data.low[0]
        c = self.data.close[0]

        # オープン中のトレード管理（エグジット判定）
        trades_to_remove = []
        for trade in self.open_trades:
            if trade.get('just_entered', False):
                trade['just_entered'] = False
                continue
            if trade.get('exited', False):
                continue

            SL = trade['SL']
            TP = trade['TP']
            exit_price = None

            if trade['direction'] == 'long':
                if o >= TP:
                    exit_price = o
                elif o <= SL:
                    exit_price = o
                elif h >= TP:
                    exit_price = TP
                elif l <= SL:
                    exit_price = SL
            elif trade['direction'] == 'short':
                if o <= TP:
                    exit_price = o
                elif o >= SL:
                    exit_price = o
                elif l <= TP:
                    exit_price = TP
                elif h >= SL:
                    exit_price = SL

            if exit_price is not None:
                trade['prices'].append(exit_price)
                trade['exited'] = True
                # 生トレードデータを記録（後で分析モジュールで処理）
                self.raw_trades.append(trade)
                trades_to_remove.append(trade)
            else:
                trade['prices'].append(c)
                trade['remaining'] -= 1
                if trade['remaining'] <= 0:
                    exit_price = c
                    trade['exited'] = True
                    self.raw_trades.append(trade)
                    trades_to_remove.append(trade)

        # 終了したトレードをオープン中リストから除外
        self.open_trades = [t for t in self.open_trades if not t.get('exited', False)]
        self.bar_count += 1

        # 新規エントリーシグナル（エントリー間隔の判定）
        if (self.bar_count - 1) % self.params.entry_interval == 0:
            if c > self.sma[0]:
                self._enter_trade(current_dt, o, direction='long')
            elif c < self.sma[0]:
                self._enter_trade(current_dt, o, direction='short')

    def _enter_trade(self, current_dt, entry_price, direction):
        if direction == 'long':
            SL = entry_price * (1 - self.params.stop_loss)
            TP = entry_price * (1 + self.params.take_profit)
        else:  # short
            SL = entry_price * (1 + self.params.stop_loss)
            TP = entry_price * (1 - self.params.take_profit)
        trade = {
            'entry_dt': current_dt,
            'entry_price': entry_price,
            'direction': direction,
            'remaining': self.params.hold_period,
            'prices': [entry_price],
            'SL': SL,
            'TP': TP,
            'exited': False,
            'just_entered': True,
        }
        self.open_trades.append(trade)
