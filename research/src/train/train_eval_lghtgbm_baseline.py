# train_eval_lghtgbm_baseline.py
#byチャッピー
from __future__ import annotations
from typing import Sequence
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd
from omegaconf import OmegaConf

def train_and_evaluate(
    cfg,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,   # ← テストは使わないので Optional
    *,
    target: str = "ret_2",
    drop_cols: Sequence[str] = (
        "vol_2", "mdd_2",
        "ret_4", "vol_4", "mdd_4",
        "ret_6", "vol_6", "mdd_6",
    ),
    val_frac: float = 0.2,                 # ← 検証サイズ (20 %)
) -> float:
    """
    train_df から時間順に (1‑val_frac):(val_frac) で分割し、
    検証 RMSE を返す。test_df は汎化確認用なので最適化には使わない。
    """

    # ---------- 前処理 (NaN / signal==0 除去) --------------------------
    train_df = (
        train_df
        .dropna(subset=[target])
        .query("signal != 0")
        .copy()
    ).sort_index()                         # 時系列順を保証

    # ---------- train / valid に時系列 split --------------------------
    split = int(len(train_df) * (1 - val_frac))
    train_part = train_df.iloc[:split]
    valid_part = train_df.iloc[split:]

    X_train = train_part.drop(columns=list(drop_cols) + [target])
    y_train = train_part[target]
    X_valid = valid_part.drop(columns=list(drop_cols) + [target])
    y_valid = valid_part[target]

    # ---------- LightGBM パラメータ -----------------------------------
    params = OmegaConf.to_container(cfg.model, resolve=True)

    # ---------- 学習 ---------------------------------------------------
    model = lgb.LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        early_stopping_rounds=200,
        verbose=False,
    )

    # ---------- 検証 RMSE ---------------------------------------------
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration_)
    val_rmse = mean_squared_error(y_valid, y_pred, squared=False)

    # ---------- （任意）テスト RMSE を確認 -----------------------------
    # if test_df is not None:
    #     test_df = (
    #         test_df
    #         .dropna(subset=[target])
    #         .query("signal != 0")
    #         .copy()
    #     )
    #     X_test = test_df.drop(columns=list(drop_cols) + [target])
    #     y_test = test_df[target]
    #     y_test_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    #     test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    #     print(f"Test RMSE (参考): {test_rmse:.4f}")

    return float(val_rmse)   # ← Optuna はこれを最小化
