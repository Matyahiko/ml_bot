#1_data_pipeline.py

from __future__ import annotations
from dataclasses import dataclass, field  
from pathlib import Path
from typing import List, Type
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline # 修正後: 明確なエイリアスを設定
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator as SklearnBaseEstimator, TransformerMixin
import joblib
import json
import traceback
from functools import reduce

import logging
from rich.logging import RichHandler

import backtrader as bt
from vectorbt.utils import checks as validation

# 以下自作関数
from transforms.data_loader import DataLoader
# 特徴量追加(非状態依存)
from transforms.stateless.add_time_features import add_time_features_cyc
from transforms.stateless.add_technical_indicators import add_technical_indicators
from transforms.stateless.add_rolling_statistics import add_rolling_statistics
from transforms.stateless.add_lag_features import add_lag_features
from transforms.stateless.drop_high_corr_features import DropHighCorrFeatures
# 特徴量追加(状態依存)
from transforms.stateful.apply_scaler import ApplyScaler

# 時系列分割関数
from validation.movingwindow_kfold import MovingWindowKFold

# ラベリング
from labels.simulator import Simulator_Vectorbt
from labels.squeeze_momentum_indicator_lb_strategy_v2 import SqueezeMomentumStrategy

# Utils
from utils.util_cache import UtilCache
from utils.flatten_df import flatten_columns

#設定
from config import PipelineConfig

# logger = logging.getLogger("data_pipeline")
# handler = RichHandler(rich_tracebacks=True)
# logger.handlers.clear()
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

# グローバルにワーカー用変数を宣言
cfg: PipelineConfig
full_pipeline = None

def _align_and_join(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """
    Align `add` to `base` on DatetimeIndex and concatenate columns.
    重複インデックスを削除し、base.index に合わせて reindex してから結合します。
    """
    add = add.loc[~add.index.duplicated()]
    add = add.reindex(base.index)
    # axis=1 で横方向にカラムを普通に結合
    return pd.concat([base, add], axis=1)

def init_worker(config: PipelineConfig):
    """
    各ワーカー起動時に一度だけ呼ばれて必要なオブジェクトを準備する
    これはグローバルにあるけどfoldを跨いでパラメータを共有はしない
    """
    global cfg, full_pipeline
    cfg = config

    # 全変換パイプライン（stateless + stateful統合版）
    full_pipeline = SklearnPipeline([
        ("time_feat", FunctionTransformer(add_time_features_cyc, validate=False)),
        ("tech_ind", FunctionTransformer(partial(add_technical_indicators, symbols=cfg.symbols, feature_lag=cfg.feature_lag, fillna=cfg.fillna, min_non_na=cfg.min_non_na), validate=False)),
        ("lag_feat", FunctionTransformer(partial(add_lag_features, lags=cfg.lag), validate=False)),
        ("roll_stat", FunctionTransformer(partial(add_rolling_statistics, windows=cfg.rolling_windows), validate=False)),
        #("drop_high_corr", DropHighCorrFeatures(threshold=cfg.corr_threshold)),
        ("scaler", ApplyScaler(scaler_name=cfg.scaler)),
    ])

def process_fold_worker(
    fold_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # パイプライン変換（fit/transformで一回だけ）
    train_t = full_pipeline.fit_transform(train_df)
    test_t = full_pipeline.transform(test_df)

    # 2) 学習済みパイプライン保存
    fold_dir = cfg.cache_dir / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipeline, fold_dir / "full_pipeline.joblib")

    # ラベリング
    from functools import reduce
    train_labels_list: list[pd.DataFrame] = []
    test_labels_list: list[pd.DataFrame] = []

    for symbol in cfg.symbols:
        if symbol not in cfg.target:
            continue
        
        simulator_train = Simulator_Vectorbt(
            price_df=train_df,
            symbol=symbol,
            strategy=cfg.strategy_cls,
            fold=fold_id
        )
        train_labels_list.append(simulator_train.get_labels())

        simulator_test = Simulator_Vectorbt(
            price_df=test_df,
            symbol=symbol,
            strategy=cfg.strategy_cls,
            fold=fold_id
        )
        test_labels_list.append(simulator_test.get_labels())

    if train_labels_list:
        combined_train = reduce(lambda l, r: l.fillna(r), train_labels_list)
        train_t = _align_and_join(train_t, combined_train)

    if test_labels_list:
        combined_test = reduce(lambda l, r: l.fillna(r), test_labels_list)
        test_t = _align_and_join(test_t, combined_test)

    # flatten_columnsの呼び出しは不要
    return {"train": train_t, "test": test_t}

class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self.dl = DataLoader(
            data_path=config.data_path,
            exchange=config.exchange,
            interval=config.interval,
            file_format=config.file_format,
        )
        self.tscv = MovingWindowKFold(
            n_splits=config.fold,
            clipping=config.clipping,
        )


    def run(self) -> None:
        df_full = self.dl.load(symbols=self.cfg.symbols)
        splits = list(self.tscv.split(df_full))
        print(f"分割完了: 分割={self.cfg.fold}, clipping={self.cfg.clipping}")

        # 並列処理 (initializer でワーカー内部にパイプラインを構築)
        with ProcessPoolExecutor(
            max_workers=self.cfg.fold,
            initializer=init_worker,
            initargs=(self.cfg,),
        ) as executor:
            futures = {
                executor.submit(
                    process_fold_worker,
                    fold,
                    df_full.iloc[train_idx],
                    df_full.iloc[test_idx],
                ): fold
                for fold, (train_idx, test_idx) in enumerate(splits)
            }
            for future in as_completed(futures):
                fold = futures[future]

                try:
                    result = future.result()
                    print(f"Fold {fold} 完了")
                    print(f"Fold {fold} データ構造: {result['train'].shape}")

                    # 保存ディレクトリの準備
                    fold_dir = self.cfg.cache_dir / f"fold{fold}"
                    fold_dir.mkdir(parents=True, exist_ok=True)

                    # Parquet形式で学習用データ保存（圧縮あり）
                    result["train"].to_parquet(fold_dir / "train.parquet", index=True, compression="snappy")
                    result["test"].to_parquet(fold_dir / "test.parquet", index=True, compression="snappy")

                    # デバッグ確認用CSV（先頭100行）
                    result["train"].head(100).to_csv(fold_dir / "train_sample.csv", index=False)
                    result["test"].head(100).to_csv(fold_dir / "test_sample.csv", index=False)

                    # オプション：メタ情報も保存（設定のスナップショット）
                    meta_info = {
                        "symbols": self.cfg.symbols,
                        "rolling_windows": self.cfg.rolling_windows,
                        "lag": self.cfg.lag,
                        "scaler": self.cfg.scaler,
                        "strategy": self.cfg.strategy_cls.__name__,
                        "fold": fold,
                    }
                    with open(fold_dir / "meta.json", "w") as f:
                        json.dump(meta_info, f, indent=4)

                except Exception as e:
                    #print(f"Error fold {fold}:{e}")
                    #print(f"Error fold {fold}: {type(e).__name__}: {e}")
                    # フルスタックを標準出力に
                    traceback.print_exc()
                    pass
                                
if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = DataPipeline(config)
    pipeline.run()
