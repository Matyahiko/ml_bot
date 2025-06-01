from __future__ import annotations
from dataclasses import dataclass, field  
from pathlib import Path
from typing import List, Type
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial  
import pandas as pd
import hashlib
import rich.logging
from rich.logging import RichHandler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import backtrader as bt

# 以下自作関数
from transforms.data_loader import DataLoader
# 特徴量追加(非状態依存)
from transforms.stateless.add_time_features import add_time_features_cyc
from transforms.stateless.add_technical_indicators import add_technical_indicators
from transforms.stateless.add_rolling_statistics import add_rolling_statistics
from transforms.stateless.add_lag_features import add_lag_features
# 特徴量追加(状態依存)
from transforms.stateful.apply_scaler import ApplyScaler, BaseEstimator

#時系列分割関数
from validation.movingwindow_kfold import MovingWindowKFold

#ラベリング
from labels.simulator import Simulator_Vectorbt
from labels.squeeze_momentum_indicator_lb_strategy_v2 import LazyBearSqueezeMomentumStrategy

#Utils
from utils.util_cache import UtilCache

logger = rich.logging.get_logger("data_pipeline")
handler = RichHandler(rich_tracebacks=True)
logger.handlers.clear()
logger.addHandler(handler)
logger.setLevel("INFO")

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
    
    #labelingパラメータ
    strategy_cls:Type[bt.Strategy] = LazyBearSqueezeMomentumStrategy
    simulator_vectorbt = Simulator_Vectorbt
    
    
class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.dl = DataLoader(
            data_path=self.cfg.data_path,
            exchange=self.cfg.exchange,
            interval=self.cfg.interval,
            file_format=self.cfg.file_format,
        )
        self.tscv = MovingWindowKFold(n_splits=self.cfg.fold, clipping=self.cfg.clipping)
        #partial???
        # ステートレス変換パイプライン
        self.stateless_pipeline = Pipeline(
            [
                ("time_feat", FunctionTransformer(add_time_features_cyc, validate=False)),
                ("tech_ind", FunctionTransformer(add_technical_indicators, validate=False)),
                (
                    "roll_stat",
                    FunctionTransformer(
                        partial(add_rolling_statistics, windows=self.cfg.rolling_windows),
                        validate=False,
                    ),
                ),
                (
                    "lag_feat",
                    FunctionTransformer(
                        partial(add_lag_features, lags=self.cfg.lag),
                        validate=False,
                    ),
                ),
            ],
        )

        # ステートフル変換 + モデル
        self.stateful_pipeline = Pipeline([
            ("scaler", ApplyScaler(scaler_name=self.cfg.scaler)),
            ("model", BaseEstimator()),
        ])
        

    def run(self) -> None:
        df_full = self.dl.load(symbol=self.cfg.symbols)
        splits = list(self.tscv.split(df_full))
        print("分割完了 分割: %d, clipping: %s", self.cfg.fold, self.cfg.clipping)

        # 並列処理
        with ProcessPoolExecutor(max_workers=self.cfg.fold) as executor:
            futures = {
                executor.submit(
                    self._process_fold,
                    fold,
                    df_full.iloc[train_idx],
                    df_full.iloc[test_idx],
                ): fold
                for fold, (train_idx, test_idx) in enumerate(splits)
            }
            for future in as_completed(futures):
                fold = futures[future]
                try:
                    future.result()
                    print(f"Fold {fold} 完了")
                except Exception:
                    print(f"Error fold: {fold}")

    def _process_fold(self, fold_id: int, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        # ステートレス変換
        train_transformed = self.stateless_pipeline.fit_transform(train_df)
        test_transformed = self.stateless_pipeline.transform(test_df)
        
        # ステートフル変換 + モデル
        train_transformed = self.stateful_pipeline.fit_transform(train_transformed)
        test_transformed = self.stateful_pipeline.transform(test_transformed)
        
        

        return {"train": train_transformed, "test": test_transformed}

if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = DataPipeline(config)
    pipeline.run()
