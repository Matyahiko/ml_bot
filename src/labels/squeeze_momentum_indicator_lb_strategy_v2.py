import vectorbt as vbt
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class SqueezeMomentumConfig:
    bb_period: int = 6           # ボリンジャー/KC の期間
    kc_multiplier: float = 0.9   # KC 用 ATR 乗数
    bb_multiplier: float = 1.2   # BB 用 σ 乗数
    linreg_period: int = 8       # モメンタム傾き計算の窓長
    stop_loss: float = 0.025     # 2.5 %
    take_profit: float = 0.03    # 3.0 %
    time_exit: int = 3           # n 本後クローズ
    threshold: float = 0.012     # |momentum| > threshold
    
class SqueezeMomentumStrategy:
    
    @staticmethod
    def _rolling_linreg_slope(arr_window: np.ndarray) -> float:
        #y = ax+bのaを返す
        n = arr_window.size
        x = np.arange(n, dtype=float)
        slope, _ = np.polyfit(x, arr_window, 1)
        return slope
    
    @staticmethod
    def _lazybear_apply(open_, high, low, close, bb_period, kc_multiplier, bb_multiplier, linreg_period):
        close = vbt.utils.validation.asarray(close)
        
        sma = vbt.MA.run(close, bb_period).ma
        std = vbt.STD.run(close, bb_period).std
        atr = vbt.ATR.run(high, low, close, bb_period).atr
        
        bb_upper = sma + bb_multiplier * std
        bb_lower = sma - bb_multiplier * std
        kc_upper = sma + kc_multiplier * atr
        kc_lower = sma - kc_multiplier * atr
        
        squeeze = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        demean = close - sma
        momentum = (
            pd.Series(demean, index=close.index)
            .rolling(linreg_period, min_periods=linreg_period)
            .apply(SqueezeMomentumStrategy._rolling_linreg_slope, raw=True)
        )

        return squeeze.astype(bool), momentum
    
    _LazyBearSMI = vbt.IndicatorFactory(
        class_name='LazyBearSMI',
        short_name='LB_SMI',
        input_names=['open', 'high', 'low', 'close'],
        param_names=['bb_period', 'kc_multiplier', 'bb_multiplier', 'linreg_period'],
        output_names=['squeeze', 'momentum'],
    ).from_apply_func(_lazybear_apply)
        
    def __init__(self, price_df: pd.DataFrame, symbol: str, config: SqueezeMomentumConfig = SqueezeMomentumConfig()):
        
        self.open  = price_df[(symbol, 'open')]
        self.high  = price_df[(symbol, 'high')]
        self.low   = price_df[(symbol, 'low')]
        self.close = price_df[(symbol, 'close')]
        
        self.ind = self._LazyBearSMI.run(
            self.open, self.high, self.low, self.close,
            bb_period=config.bb_period,
            kc_multiplier=config.kc_multiplier,
            bb_multiplier=config.bb_multiplier,
            linreg_period=config.linreg_period,
        )

        # 設定を保持
        self.cfg = config
        
    def generate_signals(self):
        squeeze   = self.ind.squeeze
        momentum  = self.ind.momentum

        released  = squeeze.shift(1) & ~squeeze #スクイーズが終わった瞬間
        strong    = squeeze & (momentum.abs() > self.cfg.threshold) #スクイーズ中に閾値越えのモメンタムを検出した場合
        trigger   = released | strong #スクイーズが終わった瞬間またはスクイーズ中に強いモメンタムが出た瞬間のどちらか

        long_entries   = trigger & (momentum > 0) #上のトリガーが発生してモメンタムが正ならロング
        short_entries  = trigger & (momentum < 0) #上のトリガーが発生してモメンタムが負ならショート

        #一定期間でイグジット
        long_time_exit  = vbt.signals.shift_n(long_entries,  self.cfg.time_exit)
        short_time_exit = vbt.signals.shift_n(short_entries, self.cfg.time_exit)
        
        #逆方向にシグナルが出ればイグジット
        long_exits  = long_time_exit  | short_entries
        short_exits = short_time_exit | long_entries

        return long_entries, long_exits, short_entries, short_exits