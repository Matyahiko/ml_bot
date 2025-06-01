#custom_analyzer.py

import backtrader as bt
import pandas as pd

class CustomAnalyzer(bt.Analyzer):
    def start(self):
        self.trade_data = []

    def next(self):
        current_dt = self.datas[0].datetime.datetime(0)
        open_price = self.datas[0].open[0]
        high = self.datas[0].high[0]
        low = self.datas[0].low[0]
        close = self.datas[0].close[0]
        volume = self.datas[0].volume[0]

        # 現在のポジション情報から取引状態を判定
        if self.strategy.position:
            trade_flag = True
            if self.strategy.position.size > 0:
                trade_type = 'long'
            elif self.strategy.position.size < 0:
                trade_type = 'short'
            else:
                trade_type = ''
        else:
            trade_flag = False
            trade_type = ''

        self.trade_data.append({
            'datetime': current_dt,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume,
            'trade_flag': trade_flag,
            'trade_type': trade_type
        })

    def stop(self):
        df = pd.DataFrame(self.trade_data)
        df.set_index('datetime', inplace=True)
        df.to_csv('storage/kline/raw_trade_data.csv')

    def get_analysis(self):
        return pd.DataFrame(self.trade_data)
