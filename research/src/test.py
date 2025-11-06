from __future__ import annotations  
from pathlib import Path
from typing import List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline 
from sklearn.preprocessing import FunctionTransformer
import joblib
import json
import traceback
import logging

# Hydra
from omegaconf import DictConfig, OmegaConf
import hydra

# Optuna
import optuna
from optuna.samplers import TPESampler

# LightGBM
from lightgbm import LGBMRegressor

# sklearn metrics
from sklearn.metrics import r2_score

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

log = logging.getLogger(__name__)

# グローバルワーカー用
cfg: DictConfig
full_pipeline: SklearnPipeline
pipeline_cfg: Any
sim_cfg: Any
strategy_cfg: Any
strategy_cls: Any
cache_dir: Path

# ---------------------- shared pipeline builder ----------------------
def build_full_pipeline(cfg: DictConfig) -> SklearnPipeline:
    """
    Hydra設定(cfg.pipeline)に基づく前処理パイプラインを返す
    """
    pcfg = cfg.pipeline
    return SklearnPipeline([
        ("time_feat", FunctionTransformer(add_time_features_cyc, validate=False)),
        ("tech_ind", FunctionTransformer(
            partial(add_technical_indicators,
                    symbols=pcfg.symbols,
                    feature_lag=pcfg.feature_lag,
                    fillna=pcfg.fillna,
                    min_non_na=pcfg.min_non_na),
            validate=False
        )),
        ("lag_feat", FunctionTransformer(
            partial(add_lag_features, lags=pcfg.lag), validate=False
        )),
        ("roll_stat", FunctionTransformer(
            partial(add_rolling_statistics, windows=pcfg.rolling_windows), validate=False
        )),
        ("scaler", ApplyScaler(scaler_name=pcfg.scaler)),
    ])

# ---------------------- batch preprocess run --------------------------
class DataPipeline:
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.cache_dir = Path(self.cfg.pipeline.cache_dir)
        self.dl = DataLoader(
            data_path=self.cfg.pipeline.data_path,
            exchange=self.cfg.pipeline.exchange,
            interval=self.cfg.pipeline.interval,
            file_format=self.cfg.pipeline.file_format,
        )
        self.tscv = MovingWindowKFold(
            n_splits=self.cfg.pipeline.fold,
            clipping=self.cfg.pipeline.clipping,
        )

    def run(self, cfg_worker: dict) -> None:
        df_full = self.dl.load(symbols=self.cfg.pipeline.symbols)
        splits = list(self.tscv.split(df_full))
        log.info(f"分割完了: fold={self.cfg.pipeline.fold}, clipping={self.cfg.pipeline.clipping}")

        # 並列処理 (initializer でパイプライン構築)
        with ProcessPoolExecutor(
            max_workers=self.cfg.pipeline.fold,
            initializer=init_worker,
            initargs=(cfg_worker,),
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
                    log.info(f"Fold {fold} 完了")
                    # 保存
                    fold_dir = self.cache_dir / f"fold{fold}"
                    fold_dir.mkdir(parents=True, exist_ok=True)
                    result["train"].to_parquet(fold_dir/"train.parquet", compression="snappy")
                    result["test"].to_parquet (fold_dir/"test.parquet", compression="snappy")
                    OmegaConf.save(config=self.cfg, f=str(fold_dir/"meta.yaml"), resolve=True)
                except Exception:
                    log.exception(f"Fold {fold} で例外発生")

# ---------------------- multiprocess worker init ----------------------
def init_worker(root_cfg: dict):
    global full_pipeline, pipeline_cfg, sim_cfg, strategy_cfg, strategy_cls, cache_dir
    cfg = OmegaConf.create(root_cfg)
    pipeline_cfg = cfg.pipeline
    sim_cfg      = cfg.simulator
    strategy_cfg = cfg.strategy
    strategy_cls = SqueezeMomentumStrategy
    cache_dir    = Path(pipeline_cfg.cache_dir)

    # ワーカー内で使うパイプライン構築
    full_pipeline = build_full_pipeline(cfg)

# ---------------------- per-fold processing --------------------------
def _align_and_join(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    add = add.loc[~add.index.duplicated()]  # 重複行除外
    add = add.reindex(base.index)
    return pd.concat([base, add], axis=1, verify_integrity=True)


def process_fold_worker(
    fold_id: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    # 前処理
    train_t = full_pipeline.fit_transform(train_df)
    test_t  = full_pipeline.transform(test_df)

    # パイプライン保存
    fold_dir = cache_dir / f"fold{fold_id}"
    joblib.dump(full_pipeline, fold_dir/"full_pipeline.joblib")

    # ラベリング
    train_labels_list, test_labels_list = [], []
    for symbol in pipeline_cfg.symbols:
        if symbol not in pipeline_cfg.target:
            continue
        sim_train = Simulator_Vectorbt(
            price_df=train_df, symbol=symbol,
            strategy_cls=strategy_cls,
            strategy_cfg=strategy_cfg,
            sim_cfg=sim_cfg, fold=fold_id,
        )
        train_labels_list.append(sim_train.get_labels())
        sim_test = Simulator_Vectorbt(
            price_df=train_df, symbol=symbol,
            strategy_cls=strategy_cls,
            strategy_cfg=strategy_cfg,
            sim_cfg=sim_cfg, fold=fold_id,
        )
        test_labels_list.append(sim_test.get_labels())

    # 結合
    if train_labels_list:
        combined_train = reduce(lambda l, r: l.fillna(r), train_labels_list)
        train_t = _align_and_join(train_t, combined_train)
    if test_labels_list:
        combined_test = reduce(lambda l, r: l.fillna(r), test_labels_list)
        test_t = _align_and_join(test_t, combined_test)

    return {"train": train_t, "test": test_t}

# ---------------------- feature search logic --------------------------
def run_feature_search(cfg: DictConfig) -> None:
    """
    Optuna を使って rolling_windows を探索し、fold4 のテストデータ上で R² を最大化する
    """
    # build base pipeline
    pipeline_cfg = cfg.pipeline
    cache_dir = Path(pipeline_cfg.cache_dir)
    fold = pipeline_cfg.fold
    fold_dir = cache_dir / f"fold{fold}"

    df_train = pd.read_parquet(fold_dir/"train.parquet")
    df_test  = pd.read_parquet(fold_dir/"test.parquet")

    # 特徴量 / 目的変数
    target_col = "ret_2"
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test  = df_test.drop(columns=[target_col])
    y_test  = df_test[target_col]

    # モデル固定パラメータ
    model_params = dict(
        objective="huber", alpha=0.9, boosting_type="gbdt",
        learning_rate=0.01, num_leaves=31, max_depth=-1,
        min_child_samples=200, subsample=0.6, colsample_bytree=0.5,
        subsample_freq=5, reg_alpha=1.0, reg_lambda=1.0,
        n_estimators=6000, random_state=42, verbosity=-1,
    )

    def objective(trial: optuna.Trial) -> float:
        # rolling_windows の探索
        rw = trial.suggest_categorical("rolling_windows", cfg.mode.feature_search.rolling_windows)
        # 再構築
        local_cfg = OmegaConf.copy(cfg)
        local_cfg.pipeline.rolling_windows = rw
        preproc = build_full_pipeline(local_cfg)
        model = LGBMRegressor(**model_params)
        pipe  = SklearnPipeline([("preproc", preproc), ("lgbm", model)])

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        return r2_score(y_test, preds)

    # Optuna Study
    study = optuna.create_study(
        direction=cfg.mode.feature_search.direction,
        sampler=TPESampler(),
    )
    study.optimize(
        objective,
        n_trials=cfg.mode.feature_search.n_trials,
        n_jobs=cfg.mode.feature_search.n_jobs,
    )

    # 結果保存
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    joblib.dump(study, "feature_search_study.pkl")

    print("Best R²:", study.best_value)
    print("Best params:", study.best_params)

# ---------------------- main dispatcher -------------------------------
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    モードに応じてバッチ前処理 or パラメータ探索を実行
    - preprocess: DataPipeline
    - feature_search: Optuna で探索
    """
    mode = cfg.mode._target_
    if mode == "preprocess":
        cfg_worker = OmegaConf.to_container(cfg, resolve=True)
        pipeline = DataPipeline(cfg)
        pipeline.run(cfg_worker)

    elif mode == "feature_search":
        run_feature_search(cfg)

    else:
        raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    main()
