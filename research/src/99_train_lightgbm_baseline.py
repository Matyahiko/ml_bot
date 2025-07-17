#!/usr/bin/env python
"""
train_lightgbm_fold4_with_validation_split.py

Train a LightGBM regression model to predict 'ret_2' on the fold-4 data.
Validation set is carved out from the training data (80/20 split).
After training with early stopping on the validation set, perform final inference on the separate test set,
and compute RMSE, R², and residual plot based on the test data.
Saves learning curve, feature importance, validation residuals, and test residuals into /app/ml_bot/tmp/fold4/.

Requirements:
  * lightgbm>=4.5.0 (pip install lightgbm --upgrade)
  * pandas, numpy, matplotlib, pyarrow
  * scikit-learn (pip install scikit-learn)

Example:
    python train_lightgbm_fold4_with_validation_split.py
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# ------------------------- main ---------------------------------------------

def main():
    # ディレクトリ設定
    data_dir = "/app/ml_bot/tmp/fold0"
    train_path = os.path.join(data_dir, "train.parquet")
    test_path = os.path.join(data_dir, "test.parquet")  # 最終評価用データ

    print("Loading data …")
    df_all = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path)

    # --- 列名サニタイズ部分を全て削除 ---

    target = "ret_2"
    drop_targets = [
        "vol_2", "mdd_2",
        "ret_4", "vol_4", "mdd_4",
        "ret_6", "vol_6", "mdd_6",
        #"signal"
    ]
    feature_cols = [c for c in df_all.columns if c not in [target] + drop_targets]

    # NaN のある行を除外
    df_all = df_all.dropna(subset=[target])
    df_test = df_test.dropna(subset=[target])
    
    df_all = df_all[df_all['signal'] != 0].copy()
    df_test = df_test[df_test['signal'] != 0].copy()
    print(f"Train shape: {df_all.shape}")
    print(f"Test shape: {df_test.shape}")

    # 学習・検証データを分割（80% train / 20% valid）
    df_train, df_valid = train_test_split(
        df_all, test_size=0.2, random_state=42
    )

    X_train, y_train = df_train[feature_cols], df_train[target]
    X_valid, y_valid = df_valid[feature_cols], df_valid[target]
    X_test, y_test   = df_test[feature_cols], df_test[target]

    # ---------------- baseline constant model RMSE on validation ----------------
    const_rmse = np.sqrt(np.mean((y_valid - y_train.mean())**2))
    print("Const-RMSE (valid):", const_rmse)

    # LightGBM datasets --------------------------------------------------------
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)

    params = dict(
        objective="huber",
        alpha=0.9,
        metric="rmse",
        boosting="gbdt",
        learning_rate=0.01,
        num_leaves=31,
        max_depth=-1,
        min_data_in_leaf=200,
        feature_fraction=0.5,
        bagging_fraction=0.6,
        bagging_freq=5,
        lambda_l1=1.0,
        lambda_l2=1.0,
        verbosity=-1,
        seed=42,
    )

    evals_result = {}

    print("Training LightGBM … (ver.", lgb.__version__, ")")
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=6000,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=1000),
            lgb.record_evaluation(evals_result),
            lgb.log_evaluation(period=100),
        ],
    )

    # ---------------- validation RMSE & R² ------------------------------------
    y_pred_valid = booster.predict(X_valid, num_iteration=booster.best_iteration)
    mask_valid = ~(np.isnan(y_pred_valid) | np.isnan(y_valid))
    rmse_valid = np.sqrt(np.mean((y_pred_valid[mask_valid] - y_valid.values[mask_valid])**2))
    ss_res = np.sum((y_valid.values[mask_valid] - y_pred_valid[mask_valid])**2)
    ss_tot = np.sum((y_valid.values[mask_valid] - y_valid.values[mask_valid].mean())**2)
    r2_valid = 1 - ss_res/ss_tot if ss_tot != 0 else float('nan')
    print(f"Validation RMSE: {rmse_valid:.6f}")
    print(f"Validation R²:   {r2_valid:.6f}")

    # 出力ディレクトリ作成
    os.makedirs(data_dir, exist_ok=True)
    
    # ---------------- output visualisations (learning & importance & valid) ---
    # 1) learning curve
    plt.figure(figsize=(8,5))
    plt.plot(evals_result['train']['rmse'], label='train')
    plt.plot(evals_result['valid']['rmse'], label='valid')
    plt.axvline(booster.best_iteration, ls='--', lw=1, label=f'best_iter={booster.best_iteration}')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title('LightGBM Training Curve (RMSE)')
    plt.legend()
    plt.savefig(os.path.join(data_dir, 'training_curve.png'), dpi=150)
    plt.close()

    # 2) feature importance
    fig, ax = plt.subplots(figsize=(10,12))
    lgb.plot_importance(
        booster,
        max_num_features=50,
        importance_type='gain',
        ax=ax,
        title='Feature Importance (gain)'  
    )
    fig.tight_layout()
    fig.subplots_adjust(left=0.30)
    fig.savefig(os.path.join(data_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3) 検証用残差プロット
    residuals_valid = y_valid.values[mask_valid] - y_pred_valid[mask_valid]
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred_valid[mask_valid], residuals_valid, alpha=0.3)
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Plot (Validation)')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'residual_plot_valid.png'), dpi=150)
    plt.close()

    # ---------------- final evaluation on test data ---------------------------
    print("Running final inference on test data …")
    y_pred_test = booster.predict(X_test, num_iteration=booster.best_iteration)
    mask_test = ~(np.isnan(y_pred_test) | np.isnan(y_test))
    if not mask_test.any():
        raise RuntimeError("All test predictions are NaN – cannot compute metrics.")
    
    rmse_test = np.sqrt(np.mean((y_pred_test[mask_test] - y_test.values[mask_test])**2))
    ss_res_test = np.sum((y_test.values[mask_test] - y_pred_test[mask_test])**2)
    ss_tot_test = np.sum((y_test.values[mask_test] - y_test.values[mask_test].mean())**2)
    r2_test = 1 - ss_res_test/ss_tot_test if ss_tot_test != 0 else float('nan')
    print(f"Test RMSE: {rmse_test:.6f}")
    print(f"Test R²:   {r2_test:.6f}")

    # ------------- 符号的中率＆混同行列 (修正版・正しい位置) ---------------
    # 予測値・正解値の符号（0:負, 1:正）に変換
    sign_true = (y_test.values[mask_test] >= 0).astype(int)
    sign_pred = (y_pred_test[mask_test] >= 0).astype(int)

    # 正負を当てた割合
    sign_acc = (sign_true == sign_pred).mean()
    print(f"Test符号的中率: {sign_acc:.4f}")

    # 混同行列 (Negative=0, Positive=1)
    cm = confusion_matrix(sign_true, sign_pred, labels=[0,1])
    print("Test符号の混同行列")
    print("        Pred-   Pred+")
    print(f"True-   {cm[0,0]:6}  {cm[0,1]:6}")
    print(f"True+   {cm[1,0]:6}  {cm[1,1]:6}")

    # テスト残差プロット
    residuals_test = y_test.values[mask_test] - y_pred_test[mask_test]
    plt.figure(figsize=(8,6))
    plt.scatter(y_pred_test[mask_test], residuals_test, alpha=0.3)
    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel('Predicted Value')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.title('Residual Plot (Test)')
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, 'residual_plot_test.png'), dpi=150)
    plt.close()

    print("Saved all plots in", data_dir)


if __name__ == "__main__":
    main()
