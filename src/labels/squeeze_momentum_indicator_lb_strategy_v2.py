import vectorbt as vbt
import numpy as np
import pandas as pd
from labels.squeeze_momentum_indicator_lb_strategy_v2_config import SqueezeMomentumConfig

class SqueezeMomentumStrategy:

    @staticmethod
    def _lazybear_apply(open_, high, low, close,
                        bb_period, kc_multiplier, bb_multiplier, linreg_period):
        sma = vbt.MA.run(close, bb_period).ma.values
        mstd = vbt.MSTD.run(close, bb_period).mstd.values
        atr  = vbt.ATR.run(high, low, close, bb_period).atr.values

        bb_upper = sma + bb_multiplier * mstd
        bb_lower = sma - bb_multiplier * mstd
        kc_upper = sma + kc_multiplier * atr
        kc_lower = sma - kc_multiplier * atr

        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        demean = close - sma

        def rolling_slope(arr, window):
            n = len(arr)
            out = np.full(n, np.nan)
            x = np.arange(window, dtype=float)
            for i in range(window - 1, n):
                y = arr[i-window+1:i+1]
                slope, _ = np.polyfit(x, y, 1)
                out[i] = slope
            return out

        momentum = rolling_slope(demean, linreg_period)
        return squeeze.astype(bool), momentum

    _LazyBearSMI = vbt.IndicatorFactory(
        class_name='LazyBearSMI',
        short_name='LB_SMI',
        input_names=['Open', 'High', 'Low', 'Close'],
        param_names=['bb_period', 'kc_multiplier', 'bb_multiplier', 'linreg_period'],
        output_names=['squeeze', 'momentum'],
    ).from_apply_func(_lazybear_apply)
        
    def __init__(self, price_df: pd.DataFrame, symbol: str,
                 config: SqueezeMomentumConfig = SqueezeMomentumConfig()):
        # ---- マルチインデックス→フラットインデックス ここを修正 ----
        self.open  = price_df[f"{symbol}_Open"]
        self.high  = price_df[f"{symbol}_High"]
        self.low   = price_df[f"{symbol}_Low"]
        self.close = price_df[f"{symbol}_Close"]
        
        self.ind = self._LazyBearSMI.run(
            self.open, self.high, self.low, self.close,
            bb_period=config.bb_period,
            kc_multiplier=config.kc_multiplier,
            bb_multiplier=config.bb_multiplier,
            linreg_period=config.linreg_period,
        )
        self.cfg = config
        
    def generate_signals(self):
        squeeze   = self.ind.squeeze
        momentum  = self.ind.momentum

        released  = squeeze.shift(1) & ~squeeze
        strong    = squeeze & (momentum.abs() > self.cfg.threshold)
        trigger   = released | strong

        long_entries   = trigger & (momentum > 0)
        short_entries  = trigger & (momentum < 0)

        long_time_exit  = long_entries.shift(self.cfg.time_exit, fill_value=False)
        short_time_exit = short_entries.shift(self.cfg.time_exit, fill_value=False)
        
        long_exits  = long_time_exit  | short_entries
        short_exits = short_time_exit | long_entries

        return long_entries, long_exits, short_entries, short_exits
