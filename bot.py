import ccxt
import time
import ta
import talib
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import backtrader as bt
from strategy import TestStrategy



def place_order(symbol, side, amount, price=None, order_type='market'):
    try:
        if order_type == 'market':
            order = bybit.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
        elif order_type == 'limit' and price is not None:
            order = bybit.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params={"time_in_force": "GTC"}
            )
        else:
            raise ValueError("無効な注文タイプまたは価格が指定されていません。")

        print(f"注文成功: {side.upper()} {amount} {symbol} at {price if price else 'Market Price'}")
        return order

    except Exception as e:
        print(f"注文エラー: {e}")
        return None



# データ取得関数
def fetch_ohlcv(symbol, timeframe='30m', limit=200):
    ohlcv = bybit.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df



def add_technical_indicators(df, prefix):
    # TA-Libでの計算
    df[f'{prefix}_DEMA'] = talib.DEMA(df['close'])
    df[f'{prefix}_EMA'] = talib.EMA(df['close'])
    df[f'{prefix}_KAMA'] = talib.KAMA(df['close'])
    df[f'{prefix}_MA'] = talib.MA(df['close'])
    df[f'{prefix}_MIDPOINT'] = talib.MIDPOINT(df['close'])
    df[f'{prefix}_SMA'] = talib.SMA(df['close'])
    df[f'{prefix}_T3'] = talib.T3(df['close'])
    df[f'{prefix}_TEMA'] = talib.TEMA(df['close'])
    df[f'{prefix}_TRIMA'] = talib.TRIMA(df['close'])
    df[f'{prefix}_WMA'] = talib.WMA(df['close'])

    # モメンタム系
    macd, macdsignal, macdhist = talib.MACD(df['close'])
    df[f'{prefix}_MACD_macd'] = macd
    df[f'{prefix}_MACD_macdsignal'] = macdsignal
    df[f'{prefix}_MACD_macdhist'] = macdhist

    df[f'{prefix}_RSI'] = talib.RSI(df['close'])
    df[f'{prefix}_MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
    df[f'{prefix}_WILLR'] = talib.WILLR(df['high'], df['low'], df['close'])
    df[f'{prefix}_ADX'] = talib.ADX(df['high'], df['low'], df['close'])
    df[f'{prefix}_CCI'] = talib.CCI(df['high'], df['low'], df['close'])
    df[f'{prefix}_OBV'] = talib.OBV(df['close'], df['volume'])

    # ボリンジャーバンド
    upper, middle, lower = talib.BBANDS(df['close'])
    df[f'{prefix}_BBANDS_upperband'] = upper
    df[f'{prefix}_BBANDS_middleband'] = middle
    df[f'{prefix}_BBANDS_lowerband'] = lower

    # ボラティリティ系
    df[f'{prefix}_ATR'] = talib.ATR(df['high'], df['low'], df['close'])
    df[f'{prefix}_NATR'] = talib.NATR(df['high'], df['low'], df['close'])
    df[f'{prefix}_STDDEV'] = talib.STDDEV(df['close'])

    # Aroon
    aroondown, aroonup = talib.AROON(df['high'], df['low'])
    df[f'{prefix}_AROON_aroondown'] = aroondown
    df[f'{prefix}_AROON_aroonup'] = aroonup
    df[f'{prefix}_AROONOSC'] = talib.AROONOSC(df['high'], df['low'])

    return df



# 戦略クラス
class TestStrategy(bt.Strategy):
    params = (
        ('atr_period', 14),  # ATRの計算期間
    )

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt} {txt}')

    def __init__(self):
        self.dataclose = self.datas[0].close

        # ラベル（0=ダウン, 1=中立, 2=アップ）
        self.predictions = self.datas[0].predictions

        # メタ予測（1ならポジションサイズを増やす）
        self.meta_predictions = self.datas[0].meta_predictions
        

        self.order = None

        # ATRを使ったボラティリティ測定（必要に応じて保持）
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)
        self.atr._minperiod = 1

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def next(self):
        # すでに注文が出ていれば、執行されるまで待機
        if self.order:
            return

        current_label = self.predictions[0]
        current_meta = self.meta_predictions[0]
        print(f'Label={current_label}, Meta={current_meta}')
        
        # メタ予測に応じた資金配分率の決定
        if current_meta == 1:
            allocation = 1
        else:
            allocation = 1

        # ポジションサイズ計算（口座残高×allocation / 現在価格）
        size = (self.broker.get_cash() * allocation) / self.dataclose[0]
        size = round(size, 8)  # 必要に応じて丸め

        # いまポジションがない場合
        if not self.position:
            # アップトレンドならロング
            if current_label == 2:
                self.order = self.buy(
                    size=size,
                    exectype=bt.Order.Market
                )
                # ろんぐ
                place_order(self.params.symbol, 'buy', self.params.order_size)
                self.log(f'LONG ENTRY (Label=2), Meta={current_meta}, Price={self.dataclose[0]:.2f}, Size={size:.8f}')

            # ダウントレンドならショート
            elif current_label == 0:
                self.order = self.sell(
                    size=size,
                    exectype=bt.Order.Market
                )
                # ショート
                place_order(self.params.symbol, 'sell', self.params.order_size)
                self.log(f'SHORT ENTRY (Label=0), Meta={current_meta}, Price={self.dataclose[0]:.2f}, Size={size:.8f}')

            # 中立（1）の場合は何もしない
            else:
                pass

        # ポジションをすでに持っている場合
        else:
            # ポジションサイズが正ならロング、負ならショート
            current_position_size = self.position.size

            # 1（中立）になったらポジションをクローズ
            if current_label == 1:
                # クローズ（マーケット注文で即決済）
                self.order = self.close(
                    exectype=bt.Order.Market
                )
                place_order(self.params.symbol, 'sell' if self.position.size > 0 else 'buy', abs(self.position.size))
                self.log(f'CLOSE (Label=1), Price={self.dataclose[0]:.2f}, Size={current_position_size:.8f}')

            # アップトレンド (2) だが今ショートを持っている場合 → 反転(ショート→ロング)
            elif current_label == 2 and current_position_size < 0:
                # まずショートポジションをクローズ
                self.order = self.close(
                    exectype=bt.Order.Market
                )
                self.log(f'CLOSE SHORT -> REVERSE TO LONG, Price={self.dataclose[0]:.2f}, Size={current_position_size:.8f}')

                # 注文完了後、次のバーでロングを作るために return して終了
                return

            # ダウントレンド (0) だが今ロングを持っている場合 → 反転(ロング→ショート)
            elif current_label == 0 and current_position_size > 0:
                # まずロングポジションをクローズ
                self.order = self.close(
                    exectype=bt.Order.Market
                )
                self.log(f'CLOSE LONG -> REVERSE TO SHORT, Price={self.dataclose[0]:.2f}, Size={current_position_size:.8f}')

                # 注文完了後、次のバーでショートを作るために return して終了
                return

            # ラベルが変わらない（同じトレンドが続いている）場合 → なにもしない（保持継続）
            #   ロング中にラベルが 2 のまま、またはショート中にラベルが 0 のまま など
            else:
                pass




def main():
    # 比嘉君のモデルで使用されているコインシンボル
    symbols = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'BCH/USDT']
    data = {}
    for symbol in symbols:
        data[symbol] = fetch_ohlcv(symbol)

    # 各通貨ペアに対して指標を追加
    for symbol in symbols:
        symbol_key = symbol.replace('/', '')
        data[symbol] = add_technical_indicators(data[symbol], symbol_key)

    # 各シンボルの最新データを取得し、1行のデータとして統合
    latest_data = {symbol: data[symbol] for symbol in symbols}

    # インデックスの重複を避けるために、各シンボルのデータを辞書形式で統合
    flattened_data = {}
    for symbol, df in latest_data.items():
        symbol_key = symbol.replace('/', '')
        flattened_data.update(df.add_prefix(f'{symbol_key}_').to_dict())

    # timestamp 列の変換
    for key in flattened_data:
        if 'timestamp' in key:
            flattened_data[key] = pd.to_datetime(flattened_data[key], unit='ms', errors='coerce')

    # DataFrameとして整形
    input_data = pd.DataFrame([flattened_data])


    normal_model = lgb.Booster(model_file=os.path.join("models","lightgbm_model_fold3.txt"))
    meta_model = lgb.Booster(model_file=os.path.join("models","lightgbm_meta_model_fold3.txt"))


    # モデルで使用する特徴量名を取得
    feature_names_normal = normal_model.feature_name()
    feature_names_nmeta = meta_model.feature_name()

    # LightGBM モデルで推論
    predictions_normal = normal_model.predict(input_data.reindex(columns=feature_names_normal))
    predictions_meta = meta_model.predict(input_data.reindex(columns=feature_names_nmeta))


    # PandasDataクラス
    class PandasData(bt.feeds.PandasData):
        lines = ('predictions', 'meta_predictions')
        params = (
            ('datetime', None),
            ('open', -1),
            ('high', -1),
            ('low', -1),
            ('close', -1),
            ('volume', -1),
            ('predictions', -1),
            ('meta_predictions', -1),
            ('backfill', True),
            ('backfill_start', True)
        )

    data_feed = PandasData(dataname=input_data)
    data_feed.predictions = predictions_normal
    data_feed.meta_predictions = predictions_meta

    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)
    cerebro.addstrategy(TestStrategy, symbol='BTC/USDT', order_size=0.001)
    cerebro.run()


    # 結果の表示
    print(f"Normal Predicted Probability: {predictions_normal}")
    print(f"Meta Predicted Probability: {predictions_meta}")


if __name__ == "__main__":

    symbol_name = 'BTC/USDT'
    order_size = 0.0001

    # bybitのtestnet用APIキーと暗号キー。好きに使いましょう。
    api_key = 'KQQn6VTieFH5hER00o'
    secret_key = 'B19sxrl2Oes6zk1aJyFnPqMIzt456UVdAuL4'


    bybit = ccxt.bybit({
        'apiKey': api_key,
        'secret': secret_key,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'} #　先物取引の場合は "linear" にする
    })

    # Testnet モードを有効化
    bybit.set_sandbox_mode(True)

    # 資産状況の確認
    sample_order = bybit.fetch_balance()
    for currency in sample_order['total']:
        total = sample_order['total'][currency]
        free = sample_order['free'][currency]
        used = sample_order['used'][currency]
        print(f"{currency}: Total={total}, Free={free}, Used={used}")

    main()
