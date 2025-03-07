import backtrader as bt
import pandas as pd
import os

class CustomAnalyzer(bt.Analyzer):
    def start(self):
        """バックテスト開始時にデータ記録用リストを初期化する"""
        self.trade_data = []

    def next(self):
        """各バーごとのデータと戦略の状態を記録する"""
        current_dt = self.datas[0].datetime.datetime(0)
        open_price = self.datas[0].open[0]
        high = self.datas[0].high[0]
        low = self.datas[0].low[0]
        close = self.datas[0].close[0]
        volume = self.datas[0].volume[0]

        # 戦略から取引フラグと取引タイプを取得
        trade_flag = self.strategy.trade_flag  # 取引中かどうかのフラグ
        trade_type = self.strategy.trade_type  # ロングかショートか

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
        """バックテスト終了時に記録したデータをCSVに出力する"""
        #これは最後のフォールドだけ残る
        self.trade_data = pd.DataFrame(self.trade_data)
        self.trade_data.set_index('datetime', inplace=True)
        self.trade_data.to_csv('storage/kline/raw_trade_data.csv')

    def get_analysis(self):
        """記録されたデータを返す"""
        return pd.DataFrame(self.trade_data)
        
        
           
