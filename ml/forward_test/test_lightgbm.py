# main.py
import os
import sys
import time
import logging
from rich import print
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt

# データ取得部分を切り出したファイルからインポート
from data_fetch import fetch_multiple_bybit_data

# モジュールのパスを追加（必要に応じて）
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.prerpocess.technical_indicators import technical_indicators
from modules.lightgbm.meta_predict import predict  # predict関数のインポート

plt.style.use("default") 
plt.rcParams["figure.figsize"] = (15,12)

# 設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = {
    'model_path': 'models/lightgbm/lightgbm_model_fold2.txt',
    'meta_model_path': None,
    'symbols': ['BTCUSDT','ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT'],  # 追加するアルトコインシンボル
    'interval': '15',     # 15分足
    'limit': 200,
    'loops': 5,           # 過去データを取得するループ回数 5で約10日分
    'cash': 1000000,
    'commission': 0.001,  # 取引手数料
    #'slippage': 0.005 
}

class BybitCSV(bt.feeds.PandasData):
    lines = ('predictions','meta_predictions')
    symbol = config['symbols'][0]
    params = (
        ('datetime', None),
        ('open', f'{symbol}_Open'),
        ('high', f'{symbol}_High'),
        ('low', f'{symbol}_Low'),
        ('close', f'{symbol}_Close'),
        ('volume', f'{symbol}_Volume'),
        ('predictions', 'predictions'),
        ('meta_predictions', 'meta_predictions'),
        ('openinterest', -1),
    )


class TestStrategy(bt.Strategy):
    params = (
        ('atr_period', 14),            # ATRの計算期間
        ('atr_stop_multiplier', 2.0),  # ATRの何倍をストップ距離とするか
        ('risk_per_trade', 0.01),      # 口座残高の何%を1トレードでリスクにさらすか (例: 1%)
        ('long_ma_period', 50),        # トレンドフィルター用の長期MA
        ('printlog', True),
    )

    def log(self, txt, dt=None):
        """ 簡易ログ出力 """
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt} {txt}')

    def __init__(self):
        # === 価格、外部からの予測 ===
        self.dataclose = self.datas[0].close
        self.predictions = self.datas[0].predictions
        self.meta_predictions = self.datas[0].meta_predictions

        # === ロング/ショート ポジション管理用 ===
        self.order = None
        self.stop_order = None
        self.entry_price = None

        # === インジケータ類 ===
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.datas[0].close,
                                                         period=self.params.long_ma_period)

    def notify_order(self, order):
        """ 注文が約定などのステータスに変化したときに呼ばれる """
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price={order.executed.price:.2f}')
                self.entry_price = order.executed.price
            elif order.issell():
                self.log(f'SELL EXECUTED, Price={order.executed.price:.2f}')

        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        """ 各バーでのロジック """
        if self.order:
            return  # 前の注文が残っているなら何もしない

        current_label = self.predictions[0]
        current_meta = self.meta_predictions[0]
        close_price = self.dataclose[0]
        atr_value = self.atr[0]

        # --- トレンドフィルター(長期MA) ---
        # 「closeが長期MAより上ならロング方向のみ / 下ならショート方向のみ」
        is_up_trend = (close_price > self.long_ma[0])
        is_down_trend = (close_price < self.long_ma[0])

        self.log(f'[Label={current_label}, Meta={current_meta}] Close={close_price:.2f} ATR={atr_value:.2f}')

        # --- ポジションサイズをリスクベースで計算 ---
        # 1トレードのリスク = atr * atr_stop_multiplier
        # 損失許容額 = 口座残高 * risk_per_trade
        # → size = 損失許容額 / (リスク幅)
        risk_distance = atr_value * self.params.atr_stop_multiplier
        cash = self.broker.get_cash()
        max_risk_amount = cash * self.params.risk_per_trade
        # (ただしrisk_distanceが0でない前提)
        size = max_risk_amount / risk_distance if risk_distance != 0 else 0
        size = round(size, 8)

        # --- エントリーロジック ---
        if not self.position:
            # uptrend + current_label == 2 → ロング
            if current_label == 2 and is_up_trend:
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                self.log(f'LONG ENTRY (Label=2), Price={close_price:.2f}, Size={size}')
                self.set_initial_stop(is_long=True, size=size)

            # downtrend + current_label == 0 → ショート
            elif current_label == 0 and is_down_trend:
                self.order = self.sell(size=size, exectype=bt.Order.Market)
                self.log(f'SHORT ENTRY (Label=0), Price={close_price:.2f}, Size={size}')
                self.set_initial_stop(is_long=False, size=size)

            else:
                pass  # 中立(1)またはトレンドフィルター不一致の場合はスルー
        else:
            # ポジションある → クロージングや反転チェック
            current_position_size = self.position.size

            # 中立になったら手仕舞い (一例)
            if current_label == 1:
                self.log(f'CLOSE (Label=1), Price={close_price:.2f}, Size={current_position_size}')
                self.close_position()
                return

            # 反転シグナル
            if current_label == 2 and current_position_size < 0:
                # ショート手仕舞い → ロング
                self.log(f'CLOSE SHORT -> REVERSE LONG, Price={close_price:.2f}')
                self.close_position(reverse=True, new_is_long=True)
                return

            if current_label == 0 and current_position_size > 0:
                # ロング手仕舞い → ショート
                self.log(f'CLOSE LONG -> REVERSE SHORT, Price={close_price:.2f}')
                self.close_position(reverse=True, new_is_long=False)
                return

            # 同方向の時はトレーリングストップ更新
            self.update_trailing_stop()

    def set_initial_stop(self, is_long=True, size=0):
        """ エントリー直後のストップ注文を出す """
        atr_value = self.atr[0]
        stop_distance = self.params.atr_stop_multiplier * atr_value

        if is_long:
            stop_price = self.dataclose[0] - stop_distance
            self.stop_order = self.sell(
                exectype=bt.Order.Stop,
                size=size,
                price=stop_price
            )
            self.log(f'SET INITIAL STOP (Long), StopPrice={stop_price:.2f}, ATR={atr_value:.2f}')
        else:
            stop_price = self.dataclose[0] + stop_distance
            self.stop_order = self.buy(
                exectype=bt.Order.Stop,
                size=size,
                price=stop_price
            )
            self.log(f'SET INITIAL STOP (Short), StopPrice={stop_price:.2f}, ATR={atr_value:.2f}')

    def update_trailing_stop(self):
        """ 動的(加速的)なトレーリングストップの簡易実装例 """
        if not self.stop_order:
            return

        current_price = self.dataclose[0]
        atr_value = self.atr[0]
        base_distance = self.params.atr_stop_multiplier * atr_value

        # --- 加速係数(簡易) ---
        # 含み益がどの程度あるかによってストップ更新をタイトにする仕組み
        # 例: (現在価格 - entry_price) / base_distance が大きいほどstopを引き上げ
        if self.entry_price is None:
            return
        pos_size = self.position.size
        if pos_size == 0:
            return

        profit_distance = abs(current_price - self.entry_price)
        # 例: 1倍以上の含み益が出たら 0.5倍に縮小、2倍ならさらに縮小...という適当な例
        accel_factor = 1.0
        if profit_distance > base_distance:
            accel_factor = 0.5
        if profit_distance > 2 * base_distance:
            accel_factor = 0.3

        # 新しいストップ距離
        new_stop_dist = base_distance * accel_factor

        # ロングの場合
        if pos_size > 0:
            desired_stop = current_price - new_stop_dist
            # もしStopOrderがそれより低かったら更新
            if self.stop_order.price < desired_stop:
                self.log(f'UPDATE STOP (Long), Old={self.stop_order.price:.2f} -> New={desired_stop:.2f}')
                self.broker.cancel(self.stop_order)
                self.stop_order = self.sell(
                    exectype=bt.Order.Stop,
                    size=pos_size,
                    price=desired_stop
                )
        # ショートの場合
        else:
            desired_stop = current_price + new_stop_dist
            if self.stop_order.price > desired_stop:
                self.log(f'UPDATE STOP (Short), Old={self.stop_order.price:.2f} -> New={desired_stop:.2f}')
                self.broker.cancel(self.stop_order)
                self.stop_order = self.buy(
                    exectype=bt.Order.Stop,
                    size=abs(pos_size),
                    price=desired_stop
                )

    def close_position(self, reverse=False, new_is_long=True):
        """ ポジションをクローズし、必要なら反転エントリーする """
        if self.stop_order:
            self.broker.cancel(self.stop_order)
            self.stop_order = None

        # クローズ
        self.order = self.close(exectype=bt.Order.Market)

        if reverse:
            # 反転エントリーの場合、現在のATRに基づく初期ストップを設定
            atr_value = self.atr[0]
            risk_distance = atr_value * self.params.atr_stop_multiplier
            cash = self.broker.get_cash()
            max_risk_amount = cash * self.params.risk_per_trade
            size = max_risk_amount / risk_distance if risk_distance != 0 else 0
            size = round(size, 8)

            if new_is_long:
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                self.log(f'REVERSE ENTRY LONG, Price={self.dataclose[0]:.2f}, Size={size}')
                self.set_initial_stop(is_long=True, size=size)
            else:
                self.order = self.sell(size=size, exectype=bt.Order.Market)
                self.log(f'REVERSE ENTRY SHORT, Price={self.dataclose[0]:.2f}, Size={size}')
                self.set_initial_stop(is_long=False, size=size)

    def stop(self):
        """ バックテスト終了時のログ """
        if self.params.printlog:
            self.log('(Params) ATR Period: %d, ATR Mult: %.2f, Risk: %.2f%%, MA: %d' %
                     (self.params.atr_period,
                      self.params.atr_stop_multiplier,
                      self.params.risk_per_trade * 100,
                      self.params.long_ma_period))


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    df_combined = pd.DataFrame()
    symbols = config['symbols']

    # --- データ取得 ---
    for sym in symbols:
        logger.info(f"Fetching data for symbol: {sym}")
        df_sym = fetch_multiple_bybit_data(
            symbol=sym,
            interval=config['interval'],
            limit=config['limit'],
            loops=config['loops']
        )
        df_sym = df_sym.add_prefix(f"{sym}_")
        df_sym = technical_indicators(df_sym, sym)
        df_combined = pd.concat([df_combined, df_sym], axis=1)

    df_combined.dropna(inplace=True)

    # --- 推論モデルによる予測 ---
    df_predictions = predict(
        df_combined, 
        config['model_path'], 
        config.get('meta_model_path')
    )

    if 'predictions' not in df_predictions.columns:
        logger.error("推論結果の 'predictions' 列が存在しません。プログラムを終了します。")
        sys.exit(1)
    df_combined['predictions'] = df_predictions['predictions']
    
    # メタモデルが存在しない場合は一律0
    if config.get('meta_model_path') and 'meta_predictions' in df_predictions.columns:
        df_combined['meta_predictions'] = df_predictions['meta_predictions']
    else:
        df_combined['meta_predictions'] = 0

    # Backtraderに投入
    forward_data = BybitCSV(dataname=df_combined)
    cerebro.adddata(forward_data, name='ForwardTest')

    # 戦略の追加
    cerebro.addstrategy(TestStrategy)

    # ブローカーの設定
    cerebro.broker.setcash(config['cash'])
    cerebro.broker.setcommission(commission=config['commission'])

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')

    # 各種アナライザーを追加
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', timeframe=bt.TimeFrame.Days, compression=1)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name='annual_return')
    cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

    results = cerebro.run()
    strat = results[0]

    final_portfolio_value = cerebro.broker.getvalue()
    initial_cash = config['cash']
    total_return_pct = ((final_portfolio_value - initial_cash) / initial_cash) * 100

    logger.info(f"最終評価額: {final_portfolio_value:.2f}")
    logger.info(f"総リターン: {total_return_pct:.2f}%")

    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
    annual_return = strat.analyzers.annual_return.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()

    if 'total' in trade_analyzer and 'closed' in trade_analyzer['total']:
        total_closed = trade_analyzer['total']['closed']
        if 'won' in trade_analyzer and 'total' in trade_analyzer['won']:
            won_total = trade_analyzer['won']['total']
        else:
            won_total = 0
        if 'lost' in trade_analyzer and 'total' in trade_analyzer['lost']:
            lost_total = trade_analyzer['lost']['total']
        else:
            lost_total = 0
        win_rate = (won_total / total_closed * 100) if total_closed > 0 else 'N/A'
    else:
        total_closed = 0
        won_total = 0
        lost_total = 0
        win_rate = 'N/A'

    report = {
        'Final Portfolio Value': final_portfolio_value,
        'Total Return (%)': total_return_pct,
        'Sharpe Ratio': sharpe.get('sharperatio', 'N/A') if sharpe else 'N/A',
        'Max Drawdown (%)': drawdown.get('max', {}).get('drawdown', 'N/A') if drawdown else 'N/A',
        'Total Trades': total_closed,
        'Won Trades': won_total,
        'Lost Trades': lost_total,
        'Win Rate (%)': win_rate,
        'Annual Return (%)': annual_return.get('yearly', {}).get('return', 'N/A') if annual_return else 'N/A',
        'SQN': sqn.get('sqn', 'N/A') if sqn else 'N/A',
    }

    logger.info("========== フォワードテストレポート ==========")
    for key, value in report.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.2f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("========================================")

    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    plot = cerebro.plot()
    fig = plot[0][0]
    fig.savefig('forward_test/forwardtest.png')
