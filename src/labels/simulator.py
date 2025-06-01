import pandas as pd
import numpy as np
import vectorbt as vbt
from dataclasses import dataclass, field  

@dataclass
class SimulatorConfig:
    freq: str = "15T"
    initial_cash: float = 100_000
    fees: float = 0.0
    slippage_bps: float = 0.0
    exec_price: str = "vwap"          # "next_open" / "mid" / "vwap" / "rand_hl"
    vwap_lookback: int = 1
    random_seed: int = 42

class Simulator_Vectorbt:
    def __init__(self, price_df: pd.DataFrame, symbol:str, strategy, strategy_kwargs: dict | None = None , simulator_config: SimulatorConfig = SimulatorConfig):
        self.cfg = simulator_config
        self.price = price_df
        self.symbol = symbol
        self.strategy_class = strategy
        self.strategy_kwargs = strategy_kwargs or {}
    
    def _make_execution_price(self) -> pd.Series:
        o = self.price[(self.symbol, "open")]
        h = self.price[(self.symbol, "high")]
        l = self.price[(self.symbol, "low")]
        c = self.price[(self.symbol, "close")]
        v = self.price[(self.symbol, "volume")]

        if self.exec_price == "next_open":
            exec_px = o.shift(-1)

        elif self.exec_price == "mid":
            exec_px = ((h + l) / 2).shift(-1)

        elif self.exec_price == "vwap":
            pv = (c * v).rolling(self.vwap_lookback).sum()
            vv = v.rolling(self.vwap_lookback).sum()
            exec_px = (pv / vv).shift(-1)

        elif self.exec_price == "rand_hl":
            low_next, high_next = l.shift(-1), h.shift(-1)
            exec_px = pd.Series(
                np.random.uniform(low_next, high_next),
                index=low_next.index
            )
        
        bps = self.cfg.slippage_bps / 1e4  # 5 bps → 0.0005
        # 売買方向で正負が変わるよう sign をあとで掛ける
        self._bps = bps
        
        return exec_px
             
    def run(self) -> vbt.Portfolio:
        strat = self.strategy_class(
            price_df=self.price,
            symbol=self.symbol,
            **self.strategy_kwargs
        )

        le, lx, se, sx = strat.generate_signals()  # long/short 4 種

        exec_px = self._make_execution_price()

        # スリッページ調整：long=+1, short=-1, flat=0
        trade_dir = le.astype(int) - se.astype(int) - lx.astype(int) + sx.astype(int)
        exec_px_adj = exec_px * (1 + trade_dir * self._bps)

        pf = vbt.Portfolio.from_signals(
            price=exec_px_adj,
            entries=le,
            exits=lx,
            short_entries=se,
            short_exits=sx,
            init_cash=self.cfg.initial_cash,
            fees=self.cfg.fees,
            freq=self.cfg.freq,
            direction="both"
        )
        return pf