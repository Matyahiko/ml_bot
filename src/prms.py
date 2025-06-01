
from pathlib import Path
from typing import List, Type
from dataclasses import dataclass, field  

@dataclass
class PipelineConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"]) 
    data_path: Path = Path("/app/ml_bot/ml/raw_data")
    exchange: str = "bybit"
    interval: str = "15m"
    file_format: str = "parquet"
    fold: int = 5
    clipping: bool = False  #falseでプリクエンシャルブロック法
    use_cache: bool = False
    cache_dir: Path = Path("/tmp/cache")

    #前処理パラメータ
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    lag: List[int] = field(default_factory=lambda: [1, 2, 3])
    scaler: str = "robust"
    
@dataclass
class SimulatorConfig:
    freq: str = "15T"
    initial_cash: float = 100_000
    fees: float = 0.0
    slippage_bps: float = 0.0
    exec_price: str = "vwap",    # "next_open" / "mid" / "vwap" / "rand_hl"
    vwap_lookback: int = 1,
    random_seed: int = 42
    