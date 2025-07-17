import pandas as pd
import numpy as np
import vectorbt as vbt
from pathlib import Path
import json

from config import SimulatorConfig

class Simulator_Vectorbt:
    def __init__(
        self,
        price_df: pd.DataFrame,
        symbol: str,
        strategy,
        fold,
        strategy_kwargs: dict | None = None,
        simulator_config: SimulatorConfig | None = None
    ):
        self.cfg = simulator_config or SimulatorConfig()
        np.random.seed(self.cfg.random_seed)
        self.price = price_df
        self.symbol = symbol
        self.fold = fold
        self.strategy_class = strategy
        self.strategy_kwargs = strategy_kwargs or {}
        self.exec_price = self.cfg.exec_price
        self.vwap_lookback = self.cfg.vwap_lookback
        self._bps = self.cfg.slippage_bps / 1e4

    def _make_execution_price(self) -> pd.Series:
        # ---- マルチインデックス→フラットインデックス ここを修正 ----
        o = self.price[f"{self.symbol}_Open"]
        h = self.price[f"{self.symbol}_High"]
        l = self.price[f"{self.symbol}_Low"]
        c = self.price[f"{self.symbol}_Close"]
        v = self.price[f"{self.symbol}_Volume"]

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
        return exec_px

    def run(self) -> vbt.Portfolio:
        strat = self.strategy_class(
            price_df=self.price,
            symbol=self.symbol,
            **self.strategy_kwargs
        )
        le, lx, se, sx = strat.generate_signals()
        exec_px = self._make_execution_price()
        trade_dir = le.astype(int) - se.astype(int) - lx.astype(int) + sx.astype(int)
        exec_px_adj = exec_px * (1 + trade_dir * self._bps)
        pf = vbt.Portfolio.from_signals(
            close=exec_px_adj,
            entries=le,
            exits=lx,
            short_entries=se,
            short_exits=sx,
            init_cash=self.cfg.initial_cash,
            fees=self.cfg.fees,
            freq=self.cfg.freq,
        )
        return pf

    def get_labels(
        self,
        horizons: int | list[int] = None,
        annualize_vol: bool = None
    ) -> pd.DataFrame:
        if horizons is None:
            horizons = self.cfg.horizons
        if annualize_vol is None:
            annualize_vol = self.cfg.annualize_vol
        if isinstance(horizons, int):
            horizons = [horizons]

        strat = self.strategy_class(
            price_df=self.price,
            symbol=self.symbol,
            **self.strategy_kwargs
        )
        long_entries, _, short_entries, _ = strat.generate_signals()
        signal = pd.Series(0, index=self.price.index, dtype="int8")
        signal[long_entries] = 1
        signal[short_entries] = -1

        exec_px = self._make_execution_price()

        labels = {"signal": signal}
        for n in horizons:
            entry_px = exec_px
            exit_px = exec_px.shift(n)
            ret_n = signal * (exit_px - entry_px) / entry_px
            labels[f"ret_{n}"] = ret_n

        label_df = pd.DataFrame(labels)
        drop_cols = [f"ret_{n}" for n in horizons]
        label_df = label_df.dropna(subset=drop_cols, how="all")
        fold_dir = self.cfg.cache_dir / f"fold{self.fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        target_cols = drop_cols
        with open(fold_dir / "target_columns.json", "w") as f:
            json.dump(target_cols, f, indent=2)
        return label_df
