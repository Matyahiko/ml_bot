# backtest_labeling.py

import os
import backtrader as bt
import pandas as pd
import numpy as np
from rich import print

from modules.squeeze_momentum_indicator_lb_strategy import LazyBearSqueezeMomentumStrategy, CustomPandasData
from modules.preprocess.trade_analysis import analyze_trades
from modules.preprocess.custom_analyzer import CustomAnalyzer


def run_backtest(df,
                 hold_period=1,
                 stop_loss=0.02,
                 take_profit=0.05,
                 start_cash=100000.0,
                 commission=0.001):
    """
    Backtraderでバックテストを実行し、Analyzerから取引データと最終ポートフォリオの価値を返す関数。
    """
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission)

    cerebro.addstrategy(
        LazyBearSqueezeMomentumStrategy,
        hold_period=hold_period,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

    data = CustomPandasData(
        dataname=df,
        timeframe=bt.TimeFrame.Minutes,
        compression=15
    )
    cerebro.adddata(data)
    
    cerebro.addanalyzer(CustomAnalyzer, _name='custom_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')

    results = cerebro.run()
    trade_data = results[0].analyzers.custom_analyzer.get_analysis()
    final_value = cerebro.broker.getvalue()
    return trade_data, final_value

def backtest_labeling_run(df_ohlc):
    """
    OHLCデータのDataFrameを用いてバックテストを実行し、
    取引解析・ラベリングを行ったDataFrameを返す関数。
    """
    df_copy = df_ohlc.copy()
    trade_data, final_value = run_backtest(
        df_copy,
        hold_period=4,
        stop_loss=0.02,
        take_profit=0.05,
        start_cash=10_000_000.0,
        commission=0.0002
    )
    print(f'Final Portfolio Value: {final_value:.2f}')
    # 取引データを解析してラベリング
    df_labeled = analyze_trades(trade_data)
    df_labeled.to_csv('storage/kline/trade_data.csv')
    #特徴量とラベルを結合
    df_labeled = pd.concat([df_copy, df_labeled], axis=1)
    
    return df_labeled

if __name__ == "__main__":
    # 実行例:
    # df_ohlc = pd.read_csv('your_ohlc_data.csv', index_col=0, parse_dates=True)
    # df_result = backtest_labeling_run(df_ohlc)
    # print(df_result.tail(20))
    pass
