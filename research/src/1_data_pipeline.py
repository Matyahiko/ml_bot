#1_data_pipeline.py

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import pandas as pd
from sklearn.pipeline import Pipeline as SklearnPipeline 
from sklearn.preprocessing import FunctionTransformer
import joblib
from functools import reduce

# Hydra
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

# 以下自作関数
from transforms.data_loader import DataLoader
# 特徴量追加(非状態依存)
from transforms.stateless.add_time_features import add_time_features_cyc
from transforms.stateless.add_technical_indicators import add_technical_indicators
from transforms.stateless.add_rolling_statistics import add_rolling_statistics
from transforms.stateless.add_lag_features import add_lag_features
# 特徴量追加(状態依存)
from transforms.stateful.apply_scaler import ApplyScaler

# 時系列分割関数
from validation.movingwindow_kfold import MovingWindowKFold

# ラベリング
from labels.simulator import Simulator_Vectorbt
from labels.squeeze_momentum_indicator_lb_strategy_v2 import SqueezeMomentumStrategy

#探索用評価モデル
from train.train_eval_lghtgbm_baseline import train_and_evaluate

# ロギング
import logging
log = logging.getLogger(__name__)


# グローバルにワーカー用変数を宣言
cfg: DictConfig
full_pipeline = None

def _align_and_join(base: pd.DataFrame, add: pd.DataFrame) -> pd.DataFrame:
    """
    Align `add` to `base` on DatetimeIndex and concatenate columns.
    重複インデックスを削除し、base.index に合わせて reindex してから結合
    """
    add = add.loc[~add.index.duplicated()]
    add = add.reindex(base.index)
    # axis=1 で横方向にカラムを普通に結合
    return pd.concat([base, add], axis=1, verify_integrity=True)

def init_worker(root_cfg: dict):
    """
    各ワーカー起動時に一度だけ呼ばれて必要なオブジェクトを準備する
    これはグローバルにあるけどfoldを跨いでパラメータを共有はしない
    渡されいるのはrootconfig
    """
    global full_pipeline, pipeline_cfg, sim_cfg, strategy_cfg, strategy_cls, cache_dir
    
    cfg = OmegaConf.create(root_cfg)
    pipeline_cfg = cfg.pipeline
    sim_cfg      = cfg.simulator
    strategy_cfg = cfg.strategy
    #TODO: 複数の戦略をテストできるように拡張予定
    strategy_cls = SqueezeMomentumStrategy
    #ワーカー内でdumpするためグローバルに持っておく
    #親でdumpすると２回シリアライズすることになるため
    cache_dir    = Path(pipeline_cfg.cache_dir)

    # 全変換パイプライン（stateless + stateful統合版）
    full_pipeline = SklearnPipeline([
        ("time_feat", FunctionTransformer(add_time_features_cyc, validate=False)),
        ("tech_ind", FunctionTransformer(partial(add_technical_indicators, symbols=pipeline_cfg.symbols,  feature_lag=pipeline_cfg.feature_lag, fillna=pipeline_cfg.fillna, min_non_na=pipeline_cfg.min_non_na), validate=False)),
        ("lag_feat", FunctionTransformer(partial(add_lag_features, lags=pipeline_cfg.lag), validate=False)),
        ("roll_stat", FunctionTransformer(partial(add_rolling_statistics, windows=pipeline_cfg.rolling_windows), validate=False)),
        ("scaler", ApplyScaler(scaler_name=pipeline_cfg.scaler)),
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
    fold_dir = cache_dir / f"fold{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_pipeline, fold_dir / "full_pipeline.joblib")

    # ラベリング
    # ラベリング用に init_worker で作ったグローバル変数を使う
    train_labels_list: list[pd.DataFrame] = []
    test_labels_list: list[pd.DataFrame] = []

    for symbol in pipeline_cfg.symbols:
        if symbol not in pipeline_cfg.target:
            continue
        
        simulator_train = Simulator_Vectorbt(
                price_df=train_df,
                symbol=symbol,
                strategy_cls=strategy_cls,
                strategy_cfg=strategy_cfg,
                sim_cfg=sim_cfg,
                fold=fold_id
        )
        train_labels_list.append(simulator_train.get_labels())

        simulator_test = Simulator_Vectorbt(
        price_df=test_df,
        symbol=symbol,
        strategy_cls=strategy_cls,
        strategy_cfg=strategy_cfg,
        sim_cfg=sim_cfg,
        fold=fold_id
        )
        test_labels_list.append(simulator_test.get_labels())

    if train_labels_list:
        combined_train = reduce(lambda l, r: l.fillna(r), train_labels_list)
        train_t = _align_and_join(train_t, combined_train)

    if test_labels_list:
        combined_test = reduce(lambda l, r: l.fillna(r), test_labels_list)
        test_t = _align_and_join(test_t, combined_test)

    return {"train": train_t, "test": test_t}

class DataPipeline:
    def __init__(self, config: DictConfig):
        self.cfg = config
        self.cache_dir = Path(self.cfg.pipeline.cache_dir)
        self.dl = DataLoader(data_path=self.cfg.pipeline.data_path, exchange=self.cfg.pipeline.exchange, interval=self.cfg.pipeline.interval, file_format=self.cfg.pipeline.file_format,)
        self.tscv = MovingWindowKFold(n_splits=self.cfg.pipeline.fold, clipping=self.cfg.pipeline.clipping,)

    def run(self, cfg_worker: dict, return_metrics: bool = False) -> dict | None:
        df_full = self.dl.load(symbols=self.cfg.pipeline.symbols)
        splits = list(self.tscv.split(df_full))
        log.info(f"分割完了: 分割={self.cfg.pipeline.fold}, clipping={self.cfg.pipeline.clipping}")
        all_scores = [] 

        # 並列処理 (initializer でワーカー内部にパイプラインを構築)
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
                    log.info(f"Fold {fold} データ構造: {result['train'].shape}")

                    fold_dir = self.cache_dir / f"fold{fold}"
                    fold_dir.mkdir(parents=True, exist_ok=True)

                    # Parquet形式で学習用データ保存（圧縮あり）
                    result["train"].to_parquet(fold_dir / "train.parquet", index=True, compression="snappy")
                    result["test"].to_parquet(fold_dir / "test.parquet", index=True, compression="snappy")

                    # デバッグ確認用CSV（先頭100行）
                    result["train"].head(100).to_csv(fold_dir / "train_sample.csv", index=False)
                    result["test"].head(100).to_csv(fold_dir / "test_sample.csv", index=False)

                    # オプション：メタ情報も保存（設定のスナップショット）
                    OmegaConf.save(config=self.cfg, f=str(fold_dir/"meta.yaml"), resolve=True)
                    
                    #パラメータの評価
                    if return_metrics:
                        score = train_and_evaluate(
                            cfg=self.cfg,
                            train_df=result["train"],           
                            test_df=result["test"],
                        )
                        all_scores.append(score)
                
                except Exception as e:
                    log.exception(f"Fold {fold} の前処理で例外発生")
                
        if return_metrics:
            avg = sum(all_scores) / len(all_scores)
            return {"average_score": avg}                    
        return None
                                
# Hydra用エントリーポイント
@hydra.main(version_base=None, config_path="conf", config_name="config")       
def main(cfg: DictConfig)-> float | None:
    #print(OmegaConf.to_yaml(cfg)) 
    # ワーカーへ渡す picklable dict
    cfg_worker = OmegaConf.to_container(cfg, resolve=True)
    pipeline = DataPipeline(cfg) 
    # hydra.job.num が存在すればスイープ
    is_sweep = HydraConfig.get().mode == "MULTIRUN"
        
    if is_sweep:
        # パラメータ探索(optuna) foldの並列で2重にならないようにn_jobsを1にする
        metrics = pipeline.run(cfg_worker, return_metrics=True)
        return float(metrics["average_score"])  
    else:
        # 通常処理
        pipeline.run(cfg_worker, return_metrics=False)
        return None
                

if __name__ == "__main__":
    main()   