# config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type
import backtrader as bt

# ラベリング用クラスのインポート
from labels.squeeze_momentum_indicator_lb_strategy_v2 import SqueezeMomentumStrategy

@dataclass
class PipelineConfig:
    # 対象シンボル
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    # ターゲットシンボル
    target:str = "BTCUSDT"

    # データ読み込み設定
    data_path: Path = Path("/app/ml_bot/data/old_data")
    exchange: str = "bybit"
    interval: str = "15m"
    file_format: str = "csv"

    # クロスバリデーション設定
    fold: int = 5
    clipping: bool = False  # False でプリクエンシャルブロック法

    # キャッシュ設定
    use_cache: bool = False
    cache_dir: Path = Path("/app/ml_bot/tmp")

    # 前処理パラメータ
    feature_lag = 1
    fillna = "ffill"             # "ffill", これは封印→"bfill", "zero", None 
    min_non_na = 0.7  
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    lag: List[int] = field(default_factory=lambda: [1, 2, 3])
    scaler: str = "robust"
    corr_threshold: float = 0.95

    # ラベリングパラメータ
    strategy_cls: Type[bt.Strategy] = SqueezeMomentumStrategy

@dataclass
class SimulatorConfig:
    freq: str = "15T"
    initial_cash: float = 100_000
    fees: float = 0.0
    slippage_bps: float = 0.0
    exec_price: str = "vwap" # "next_open" / "mid" / "vwap" / "rand_hl"
    vwap_lookback: int = 1
    random_seed: int = 42
    horizons: int | list[int] = field(default_factory=lambda: [2, 4, 6]) # 例: 24 あるいは [24, 48, 96] 保有期間
    annualize_vol: bool = False # σ を √n 倍するか
    cache_dir: Path = Path("/app/ml_bot/tmp")
