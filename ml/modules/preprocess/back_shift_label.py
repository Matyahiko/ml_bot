import backtrader as bt
import pandas as pd
import numpy as np
from rich import print

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


class BetterStrategy(bt.Strategy):
    params = (
        ('fast_period', 10),       # 短期移動平均期間
        ('slow_period', 30),       # 長期移動平均期間
        ('risk_per_trade', 0.02),  # 1トレードあたりのリスク割合
        ('rsi_period', 14),        # RSI の計算期間
        ('atr_period', 14),        # ATR の計算期間
        ('atr_stop_mul', 1.5),     # ATR の何倍をストップに使うか
        ('atr_tp_mul', 2.0),       # ATR の何倍を利確ラインに使うか
    )
    
    def __init__(self):
        # 移動平均線
        self.fastma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.fast_period)
        self.slowma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.slow_period)
        
        # クロスオーバーシグナル
        self.crossover = bt.indicators.CrossOver(self.fastma, self.slowma)
        
        # RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

        # ATR
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)

        self.order = None
        
        # エントリー時の情報
        self.entry_price = None
        self.entry_dt = None
        self.trade_type = None
        
        # バックテスト結果を保持するリスト
        #   - trade_labels : (entry_dt, exit_dt, label)  最終損益ラベル(勝ち=1/負け=0)
        #   - open_position_bars : ポジション保有中バー (dt, 1)
        self.trade_labels = []
        self.open_position_bars = []
    
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED at {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'SELL EXECUTED at {order.executed.price:.2f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        
        self.order = None
    
    def next(self):
        """ 
        1ポジションのみ:
          - ポジションがない場合にエントリー判定
          - ポジションがある場合、ATRベースのストップ/リミット or 移動平均反転 でクローズ
          - ポジション保有中のバーを (dt, 1) として記録
        """
        if self.order:
            return
        
        atr_value = self.atr[0]  # 現時点での ATR

        # ポジション保有中なら、そのバーを1とする
        if self.position:
            bar_dt = self.data.datetime.datetime(0)
            self.open_position_bars.append((bar_dt, 1))
            
            # --- ATRベースのストップ or リミット判定 ---
            if self.position.size > 0:
                # ロングの場合
                # 損切りライン
                stop_price = self.entry_price - self.params.atr_stop_mul * atr_value
                # 利確ライン
                tp_price = self.entry_price + self.params.atr_tp_mul * atr_value

                # 損切り発動
                if self.data.close[0] < stop_price:
                    self.log(f'CLOSE LONG (Stop) at {self.data.close[0]:.2f}')
                    self.order = self.close()
                # 利確発動
                elif self.data.close[0] > tp_price:
                    self.log(f'CLOSE LONG (TakeProfit) at {self.data.close[0]:.2f}')
                    self.order = self.close()
                # もし「移動平均クロスが反転したら手仕舞う」というロジックを加えたければ:
                # elif self.crossover[0] < 0:
                #     self.log(f'CLOSE LONG (MA cross) at {self.data.close[0]:.2f}')
                #     self.order = self.close()

            elif self.position.size < 0:
                # ショートの場合
                stop_price = self.entry_price + self.params.atr_stop_mul * atr_value
                tp_price = self.entry_price - self.params.atr_tp_mul * atr_value

                if self.data.close[0] > stop_price:
                    self.log(f'CLOSE SHORT (Stop) at {self.data.close[0]:.2f}')
                    self.order = self.close()
                elif self.data.close[0] < tp_price:
                    self.log(f'CLOSE SHORT (TakeProfit) at {self.data.close[0]:.2f}')
                    self.order = self.close()
                # もしMA反転でクローズしたいなら以下のような条件追加
                # elif self.crossover[0] > 0:
                #     self.log(f'CLOSE SHORT (MA cross) at {self.data.close[0]:.2f}')
                #     self.order = self.close()

        # まだポジションが無い場合 → エントリー条件
        else:
            # 移動平均クロス & RSI フィルタ
            # 例）クロスが上向き かつ RSI > 55 のときロング
            #     クロスが下向き かつ RSI < 45 のときショート
            if self.crossover[0] > 0 and self.rsi[0] > 55:
                size = self.size_calc()
                self.order = self.buy(size=size)
                self.entry_price = self.data.close[0]
                self.entry_dt = self.data.datetime.datetime(0)
                self.trade_type = 'long'
                self.log(f'LONG ENTRY at {self.data.close[0]:.2f}, size={size}')

            elif self.crossover[0] < 0 and self.rsi[0] < 45:
                size = self.size_calc()
                self.order = self.sell(size=size)
                self.entry_price = self.data.close[0]
                self.entry_dt = self.data.datetime.datetime(0)
                self.trade_type = 'short'
                self.log(f'SHORT ENTRY at {self.data.close[0]:.2f}, size={size}')
    
    def notify_trade(self, trade):
        """
        トレードがクローズした時点で「最終損益ラベル」を決定し、
        (entry_dt, exit_dt, label) を記録する。
        """
        if trade.isclosed and self.entry_price is not None and self.entry_dt is not None:
            exit_dt = self.data.datetime.datetime(0)
            exit_price = trade.price  # 決済時の平均価格
            
            if self.trade_type == 'long':
                profit_rate = (exit_price - self.entry_price) / self.entry_price
            elif self.trade_type == 'short':
                profit_rate = (self.entry_price - exit_price) / self.entry_price
            else:
                profit_rate = 0
            
            label = 1 if profit_rate > 0 else 0
            
            self.log(f'TRADE CLOSED at {exit_price:.2f} | Profit Rate: {profit_rate:.4f} -> Label: {label}')
            
            # (エントリーバー日時, イグジットバー日時, 勝敗ラベル)
            self.trade_labels.append((self.entry_dt, exit_dt, label))
            
            # リセット
            self.entry_price = None
            self.entry_dt = None
            self.trade_type = None

    def size_calc(self):
        """単純な資金管理例: 現在のキャッシュの risk_per_trade 分を利用"""
        cash = self.broker.getcash()
        risk_amount = cash * self.params.risk_per_trade
        size = int(risk_amount / self.data.close[0])
        return max(size, 1)

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt} {txt}')


##################################
# バックテスト実行＋教師ラベル生成用関数
##################################
def run_backtest_and_generate_labels_back_shift(df):
    """
    - 1ポジションのみ可変エントリー・イグジット
    - ポジション保有中のバーに label=1 を付与
    - 最終的に負けトレードなら、エントリーからイグジットまで 0 に上書き
    - ポジション外は NaN のまま
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)  # 初期資金
    cerebro.addstrategy(BetterStrategy)

    data = CustomPandasData(dataname=df, timeframe=bt.TimeFrame.Minutes, compression=15)
    cerebro.adddata(data)
    cerebro.broker.setcommission(commission=0.0002)

    strategies = cerebro.run()
    strategy = strategies[0]
    final_value = cerebro.broker.getvalue()

    # ラベル用カラムを NaN で初期化
    df['label_trade_back_shift'] = np.nan

    # 1) ポジション保有中のバーを先に 1 にセット
    for dt_open, label_open in strategy.open_position_bars:
        ts_open = pd.Timestamp(dt_open)
        if ts_open in df.index:
            df.loc[ts_open, 'label_trade_back_shift'] = label_open  # 1
        else:
            print(f'Warning: 日付 {ts_open} は df のインデックスに見つかりませんでした。(open_position)')

    # 2) 負けトレード(=0) の場合は、エントリー～イグジットまで 0 に上書き
    for entry_dt, exit_dt, label_final in strategy.trade_labels:
        ts_entry = pd.Timestamp(entry_dt)
        ts_exit = pd.Timestamp(exit_dt)

        if ts_entry not in df.index:
            print(f'Warning: entry_dt {ts_entry} not in df index.')
            continue

        if ts_exit not in df.index:
            print(f'Warning: exit_dt {ts_exit} not in df index, partial fill might occur.')
        
        if label_final == 0:
            if ts_entry <= ts_exit:
                df.loc[ts_entry:ts_exit, 'label_trade_back_shift'] = 0
            else:
                df.loc[ts_exit:ts_entry, 'label_trade_back_shift'] = 0
        # label_final == 1 の場合は、最初に付けた 1 を維持
        
    df['label_trade_back_shift'].fillna(2, inplace=True)  # NaN は 2 に置換

    return df, final_value
