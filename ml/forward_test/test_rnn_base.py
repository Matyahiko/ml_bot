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

# --- RNN 用のインポート ---
from modules.rnn_base.rnn_data_process import TimeSeriesDataset
from modules.rnn_base.rnn_base import SimpleGRUModel

plt.style.use("default")
plt.rcParams["figure.figsize"] = (15, 12)

# 設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

config = {
    'model_path': 'models/rnn_base/rnn_model_fold3.pt',  # 学習済み RNN チェックポイントのパス（state_dict と hyperparameters を含む）
    'symbols': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT'],         # 追加するアルトコインシンボル
    'interval': '15',     # 15分足
    'limit': 200,
    'loops': 5,           # 過去データを取得するループ回数（例: 5で約10日分）
    'cash': 1000000,
    'commission': 0.0002,  # 取引手数料
    'device': 'cuda:0',   # 利用するデバイス（例: 'cuda:0' または 'cpu'）
    'target_column': 'target',  # RNN のデータセット作成時に利用するターゲット列の名前
}


# ============================
# RNN 推論用の関数 (LightGBM の predict の代替)
# ============================
def rnn_predict(df, model_path, batch_size=32, device="cuda:0", target_column="target"):
    """
    Parameters
    ----------
    df : pandas.DataFrame
        推論対象のデータフレーム（ターゲット列は存在しない前提）
    model_path : str
        学習済み RNN モデルのチェックポイントのパス（state_dict とハイパーパラメータ を含む）
    batch_size : int
        推論時のバッチサイズ
    device : str
        利用するデバイス
    target_column : str
        データセット作成時に必要なターゲット列の名称（推論時には使われません）
        
    Returns
    -------
    df_pred : pandas.DataFrame
        各ウィンドウの最終行に対応するインデックスを持つ "predictions" 列を持つ DataFrame
    """
    import torch
    # まずチェックポイントを読み込み、保存した state_dict とハイパーパラメータを復元
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['state_dict']
    hyperparams = checkpoint.get('hyperparameters', {})
    
    # チェックポイントに保存されたウィンドウサイズを利用（なければデフォルト405）
    window_size = hyperparams.get('window_size', 405)
    
    # 推論用データセット作成：predict_df に df を渡す
    dataset = TimeSeriesDataset(
        predict_df=df, 
        target_column=target_column, 
        window_size=window_size, 
        stride=1
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # サンプルから入力特徴量の次元を取得
    sample = next(iter(dataloader))
    feature_dim = sample.shape[-1]
    print(f"入力特徴量の次元: {feature_dim}")
    
    # チェックポイントから復元したハイパーパラメータを用いてモデルを構築
    hidden_size = hyperparams.get('hidden_size', 128)
    num_layers = hyperparams.get('num_layers', 1)
    dropout_rate = hyperparams.get('dropout_rate', 0.0)
    model = SimpleGRUModel(
        input_size=feature_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        dropout_rate=dropout_rate
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        # 推論用データセットは __getitem__ で特徴量のみを返す
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # 例として出力の argmax を予測ラベルとする
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())

    # ウィンドウ作成時のループでは各ウィンドウの予測は、
    # 入力データのインデックス i + window_size - 1 の位置に対応しているため、
    # そのインデックスを計算する
    ws = dataset.window_size
    st = dataset.stride
    window_indices = [df.index[i + ws - 1] for i in range(0, len(df) - ws, st)]

    if len(window_indices) != len(predictions):
        print(f"警告: ウィンドウの数 ({len(window_indices)}) と予測数 ({len(predictions)}) が一致しません。")

    df_pred = pd.DataFrame({"predictions": predictions}, index=window_indices)
    return df_pred


# ============================
# Backtrader 用のデータフィード定義
# ============================
class BybitCSV(bt.feeds.PandasData):
    lines = ('predictions', 'meta_predictions')
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


# ============================
# Backtrader 戦略
# ============================
class TestStrategy(bt.Strategy):
    params = (
        ('atr_period', 14),
        ('atr_stop_multiplier', 2.0),
        ('risk_per_trade', 0.1),
        ('long_ma_period', 50),
        ('printlog', True),
    )

    def log(self, txt, dt=None):
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print(f'{dt} {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.predictions = self.datas[0].predictions
        self.meta_predictions = self.datas[0].meta_predictions

        self.order = None
        self.stop_order = None
        self.entry_price = None

        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.datas[0].close,
                                                         period=self.params.long_ma_period)

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price={order.executed.price:.2f}')
                self.entry_price = order.executed.price
            elif order.issell():
                self.log(f'SELL EXECUTED, Price={order.executed.price:.2f}')
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        if self.order:
            return

        current_label = self.predictions[0]
        current_meta = self.meta_predictions[0]
        close_price = self.dataclose[0]
        atr_value = self.atr[0]

        is_up_trend = (close_price > self.long_ma[0])
        is_down_trend = (close_price < self.long_ma[0])

        self.log(f'[Label={current_label}, Meta={current_meta}] Close={close_price:.2f} ATR={atr_value:.2f}')

        risk_distance = atr_value * self.params.atr_stop_multiplier
        cash = self.broker.get_cash()
        max_risk_amount = cash * self.params.risk_per_trade
        size = max_risk_amount / risk_distance if risk_distance != 0 else 0
        size = round(size, 8)

        if not self.position:
            if current_label == 2 and is_up_trend:
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                self.log(f'LONG ENTRY (Label=2), Price={close_price:.2f}, Size={size}')
                self.set_initial_stop(is_long=True, size=size)
            elif current_label == 0 and is_down_trend:
                self.order = self.sell(size=size, exectype=bt.Order.Market)
                self.log(f'SHORT ENTRY (Label=0), Price={close_price:.2f}, Size={size}')
                self.set_initial_stop(is_long=False, size=size)
        else:
            current_position_size = self.position.size
            if current_label == 1:
                self.log(f'CLOSE (Label=1), Price={close_price:.2f}, Size={current_position_size}')
                self.close_position()
                return
            if current_label == 2 and current_position_size < 0:
                self.log(f'CLOSE SHORT -> REVERSE LONG, Price={close_price:.2f}')
                self.close_position(reverse=True, new_is_long=True)
                return
            if current_label == 0 and current_position_size > 0:
                self.log(f'CLOSE LONG -> REVERSE SHORT, Price={close_price:.2f}')
                self.close_position(reverse=True, new_is_long=False)
                return
            self.update_trailing_stop()

    def set_initial_stop(self, is_long=True, size=0):
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
        if not self.stop_order:
            return

        current_price = self.dataclose[0]
        atr_value = self.atr[0]
        base_distance = self.params.atr_stop_multiplier * atr_value

        if self.entry_price is None:
            return
        pos_size = self.position.size
        if pos_size == 0:
            return

        profit_distance = abs(current_price - self.entry_price)
        accel_factor = 1.0
        if profit_distance > base_distance:
            accel_factor = 0.5
        if profit_distance > 2 * base_distance:
            accel_factor = 0.3

        new_stop_dist = base_distance * accel_factor

        if pos_size > 0:
            desired_stop = current_price - new_stop_dist
            if self.stop_order.price < desired_stop:
                self.log(f'UPDATE STOP (Long), Old={self.stop_order.price:.2f} -> New={desired_stop:.2f}')
                self.broker.cancel(self.stop_order)
                self.stop_order = self.sell(
                    exectype=bt.Order.Stop,
                    size=pos_size,
                    price=desired_stop
                )
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
        if self.stop_order:
            self.broker.cancel(self.stop_order)
            self.stop_order = None

        self.order = self.close(exectype=bt.Order.Market)

        if reverse:
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
    
    df_combined.to_csv('forward_test/forwardtest.csv')

    # --- RNN 推論による予測 ---
    df_predictions = rnn_predict(
        df_combined,
        config['model_path'],
        batch_size=32,
        device=config['device'],
        target_column=config['target_column']
    )

    if 'predictions' not in df_predictions.columns:
        logger.error("推論結果の 'predictions' 列が存在しません。プログラムを終了します。")
        sys.exit(1)
    df_combined['predictions'] = df_predictions['predictions']

    # meta_predictions が不要の場合は一律 0 を設定
    df_combined['meta_predictions'] = 0

    # Backtrader に投入
    forward_data = BybitCSV(dataname=df_combined)
    cerebro.adddata(forward_data, name='ForwardTest')

    cerebro.addstrategy(TestStrategy)

    cerebro.broker.setcash(config['cash'])
    cerebro.broker.setcommission(commission=config['commission'])

    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')

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

    # --- ここから利益損失のプロット作成 ---
    # TimeReturn アナライザーから取得した各日の累積リターン（＝ (ポートフォリオ値／初期資金) - 1 ）の辞書
    time_return = strat.analyzers.time_return.get_analysis()
    # 辞書を pandas Series に変換（キーは日付）
    time_return_series = pd.Series(time_return)
    time_return_series.index = pd.to_datetime(time_return_series.index)
    time_return_series.sort_index(inplace=True)

    # 累積の利益損失（絶対値）を計算
    cumulative_profit = time_return_series * config['cash']
    # 日毎の利益損失は前日との差分（初日の損益はその日の累積損益）
    daily_profit = cumulative_profit.diff().fillna(cumulative_profit.iloc[0])

    fig2, ax2 = plt.subplots(figsize=(15, 12))
    ax2.bar(daily_profit.index, daily_profit, color='skyblue', label='Daily Profit/Loss')
    ax2.plot(cumulative_profit.index, cumulative_profit, color='red', marker='o', linewidth=2, label='Cumulative Profit/Loss')
    ax2.set_title('Daily and Cumulative Profit/Loss')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Profit/Loss')
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    profit_loss_file = 'forward_test/profit_loss.png'
    fig2.savefig(profit_loss_file)
    logger.info(f"Profit/Loss plot saved to {profit_loss_file}")
    # --- ここまで利益損失のプロット作成 ---

    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')

    plot = cerebro.plot()
    fig = plot[0][0]
    fig.savefig('forward_test/forwardtest.png')