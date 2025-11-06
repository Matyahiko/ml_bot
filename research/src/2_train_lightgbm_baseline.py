#!/usr/bin/env python
"""
2_train_lightgbm_pipeline.py

Train a LightGBM regression model wrapped in an sklearn Pipeline to predict 'ret_2'.
The preprocessing steps are defined in data_pipeline.build_preprocessor(), which reads
Hydra configs from conf/, and returns an sklearn Transformer. After training with early
stopping, this script produces:

  * Validation & test RMSE, R²
  * Sign-accuracy & confusion matrix on the test set
  * Learning curve, feature importance, and residual plots
    saved under /app/ml_bot/tmp/fold0/

Usage:
    python 2_train_lightgbm_pipeline.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from lightgbm import LGBMRegressor, plot_importance

# 前処理パイプラインを返す関数を import
from data_pipeline import build_preprocessor


def main():
    # --- 設定 -------------------------------------------------------
    data_dir = "/app/ml_bot/tmp/fold0"
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.parquet")
    test_path  = os.path.join(data_dir, "test.parquet")

    # --- データ読み込み ----------------------------------------------
    print("Loading data …")
    df_all  = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # --- ターゲット・特徴量選定 & 前処理不要列除外 ---------------
    target     = "ret_2"
    drop_cols  = ["vol_2","mdd_2","ret_4","vol_4","mdd_4","ret_6","vol_6","mdd_6"]
    feat_cols  = [c for c in df_all.columns if c not in [target] + drop_cols]

    # NaN や signal==0 を除外
    df_all  = df_all .dropna(subset=[target]).query("signal != 0")
    df_test = df_test.dropna(subset=[target]).query("signal != 0")

    X_all, y_all       = df_all[feat_cols], df_all[target]
    X_test, y_test     = df_test[feat_cols], df_test[target]

    # 学習・検証データ分割 (80% train / 20% valid)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )
    print(f"Train shape: {X_train.shape}, Valid shape: {X_val.shape}, Test shape: {X_test.shape}")

    # --- Pipeline 定義 -----------------------------------------------
    preprocessor = build_preprocessor()  # ColumnTransformer / Pipeline を返す
    model = LGBMRegressor(
        objective="huber",
        alpha=0.9,
        boosting_type="gbdt",
        learning_rate=0.01,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=200,
        subsample=0.6,
        colsample_bytree=0.5,
        subsample_freq=5,
        reg_alpha=1.0,
        reg_lambda=1.0,
        n_estimators=6000,
        random_state=42,
        verbosity=-1,
    )

    pipeline = Pipeline([
        ("preproc", preprocessor),
        ("lgbm", model),
    ])

    # --- 学習 (early stopping) ----------------------------------------
    print("Training Pipeline with early stopping …")
    pipeline.fit(
        X_train, y_train,
        lgbm__eval_set=[(X_val, y_val)],
        lgbm__eval_metric="rmse",
        lgbm__early_stopping_rounds=1000,
        lgbm__verbose=100,
    )

    # --- 検証評価 ----------------------------------------------------
    y_val_pred = pipeline.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)
    r2_val   = r2_score(y_val, y_val_pred)
    print(f"Validation RMSE: {rmse_val:.6f}")
    print(f"Validation R²:   {r2_val:.6f}")

    # --- テスト評価 --------------------------------------------------
    y_test_pred = pipeline.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
    r2_test   = r2_score(y_test, y_test_pred)
    print(f"Test RMSE: {rmse_test:.6f}")
    print(f"Test R²:   {r2_test:.6f}")

    # 符号的中率 & 混同行列
    sign_true = (y_test >= 0).astype(int)
    sign_pred = (y_test_pred >= 0).astype(int)
    sign_acc = (sign_true == sign_pred).mean()
    print(f"Test Sign Accuracy: {sign_acc:.4f}")
    print("Confusion Matrix (0=Negative, 1=Positive):")
    print(confusion_matrix(sign_true, sign_pred, labels=[0,1]))

    # --- 可視化出力 ----------------------------------------------
    # evals_result_ はモデル内に格納されている
    evals = pipeline.named_steps["lgbm"].evals_result_

    # 1) Learning Curve
    plt.figure(figsize=(8, 5))
    plt.plot(evals["training"]["rmse"], label="train")
    plt.plot(evals["valid_1"]["rmse"],    label="valid")
    best_iter = pipeline.named_steps["lgbm"].best_iteration_
    plt.axvline(best_iter, ls="--", lw=1, label=f"best_iter={best_iter}")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("LightGBM Training Curve (RMSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "training_curve.png"), dpi=150)
    plt.close()

    # 2) Feature Importance (gain)
    fig, ax = plt.subplots(figsize=(10, 12))
    plot_importance(
        pipeline.named_steps["lgbm"].booster_,
        max_num_features=50,
        importance_type="gain",
        ax=ax,
        title="Feature Importance (gain)",
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.30)
    fig.savefig(os.path.join(data_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) Residual Plot (Validation)
    res_val = y_val.values - y_val_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_val_pred, res_val, alpha=0.3)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot (Validation)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "residual_plot_valid.png"), dpi=150)
    plt.close()

    # 4) Residual Plot (Test)
    res_test = y_test.values - y_test_pred
    plt.figure(figsize=(8,6))
    plt.scatter(y_test_pred, res_test, alpha=0.3)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title("Residual Plot (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "residual_plot_test.png"), dpi=150)
    plt.close()

    print(f"All plots saved to {data_dir}")


if __name__ == "__main__":
    main()
