#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
meta_model_lightgbm_updated.py

【概要】
前回作成したメタラベル付きデータセット (bybit_BTCUSDT_15m_data_with_meta_labels.pkl) を用いて、
最新の LightGBM 記法（lgb.train とコールバックの利用）によりメタモデルを学習します。
今回は、クラス不均衡対策として学習データにダウンサンプリングを実施し、
学習曲線、ROC Curve、特徴量寄与率の画像を plots/meta_lightgbm に保存します。

【手順】
1. データセットの読み込みおよび前処理
2. 学習用特徴量とターゲットの定義
3. 学習データと検証データに分割し、学習データに対してダウンサンプリングを実施
4. lgb.Dataset の作成
5. 学習パラメータおよびコールバックの定義
6. lgb.train による学習実行
7. 性能評価（混同行列、教師ラベルの割合の出力）および学習曲線の保存
8. ROC Curve のプロットと保存（ROC AUC の評価画像）
9. 特徴量寄与率のプロットと保存
10. 学習済みモデル・評価結果の保存
"""

import os
import pandas as pd
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

def main():
    # ① データセットの読み込み
    data_path = "storage/kline/bybit_BTCUSDT_15m_data_with_meta_labels.pkl"
    print(f"[INFO] データ読み込み: {data_path}")
    df = pd.read_pickle(data_path)
    print(f"[INFO] データサイズ: {df.shape}")
    print(f"[INFO] 先頭5行:\n{df.head()}")

    # meta_label が存在する行のみ利用（欠損値除去）
    df = df.dropna(subset=["meta_label"])

    # ② 特徴量とターゲットの定義
    # 例として、BTCUSDT の OHLCV と RNN 予測結果（predictions）を利用
    feature_columns = []
    for col in ['BTCUSDT_Open', 'BTCUSDT_High', 'BTCUSDT_Low', 'BTCUSDT_Close', 'BTCUSDT_Volume', 'log_return', 'vol_predictions']:
        if col in df.columns:
            feature_columns.append(col)

    if not feature_columns:
        raise ValueError("利用可能な特徴量が見つかりません。データ内容を確認してください。")

    X = df[feature_columns].copy()
    y = df["meta_label"]

    # インデックスが DatetimeIndex の場合、日時から曜日・時刻の特徴量を追加
    if isinstance(df.index, pd.DatetimeIndex):
        X["dayofweek"] = df.index.dayofweek   # 0:月曜～6:日曜
        X["hour"] = df.index.hour             # 時刻（0～23）
        feature_columns.extend(["dayofweek", "hour"])

    print("[INFO] 使用する特徴量:", feature_columns)

    # ③ 学習用／検証用データに分割（80% 学習、20% 検証；層化抽出）
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"[INFO] 分割後の学習データサイズ: {X_train.shape}, 検証データサイズ: {X_val.shape}")

    # ③-2 ダウンサンプリング実施：学習データの多数派クラス（0）をマイノリティクラス（1）のサンプル数に合わせる
    train_df = pd.concat([X_train, y_train], axis=1)
    df_majority = train_df[train_df['meta_label'] == 0]
    df_minority = train_df[train_df['meta_label'] == 1]
    
    n_minority = len(df_minority)
    print(f"[INFO] マイノリティクラス（1）のサンプル数: {n_minority}")
    
    # 多数派クラスからランダムに n_minority 件を抽出
    df_majority_downsampled = df_majority.sample(n=n_minority, random_state=42)
    
    # ダウンサンプリング後の学習データの作成（シャッフルして結合）
    balanced_train_df = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)
    X_train_balanced = balanced_train_df.drop(columns=['meta_label'])
    y_train_balanced = balanced_train_df['meta_label']
    print(f"[INFO] ダウンサンプリング後の学習データサイズ: {X_train_balanced.shape}")

    # ④ lgb.Dataset の作成（ダウンサンプリング後のデータを使用）
    lgb_train = lgb.Dataset(X_train_balanced, label=y_train_balanced)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # ⑤ 学習パラメータの定義（scale_pos_weight は不要）
    params = {
        'objective': 'binary',      # 2値分類
        'metric': 'auc',
        'boosting': 'gbdt',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'verbose': -1,
        'seed': 42,
    }

    # 評価結果を記録するための辞書
    evals_result = {}

    # ⑥ コールバックの定義（早期終了、ログ出力、評価記録）
    callbacks = [
        lgb.record_evaluation(evals_result),
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]

    # ⑦ 学習の実行（最新の記法：lgb.train を使用）
    print("[INFO] LightGBM モデルの学習開始")
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=250,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'validation'],
        callbacks=callbacks
    )

    # ⑧ モデル評価
    # 検証データに対する予測（確率値）
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    # 0.5 を閾値として 2 値ラベルに変換
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)
    print(f"[評価結果] Accuracy: {acc:.4f}, ROC AUC: {auc:.4f}")

    # 混同行列の計算と出力
    cm = confusion_matrix(y_val, y_pred)
    print("[評価結果] 混同行列:")
    print(cm)

    # 教師ラベル（meta_label）の割合の表示（検証データ）
    teacher_label_ratio = y_val.value_counts(normalize=True)
    print("[評価結果] 検証データにおける教師ラベルの割合:")
    print(teacher_label_ratio)

    # プロットディレクトリの作成
    plot_dir = "plots/meta_lightgbm"
    os.makedirs(plot_dir, exist_ok=True)

    # ⑨ 学習曲線のプロットと保存（AUC の推移）
    if "auc" in evals_result["train"]:
        train_auc = evals_result["train"]["auc"]
        val_auc = evals_result["validation"]["auc"]

        plt.figure(figsize=(10, 6))
        plt.plot(train_auc, label="Train AUC")
        plt.plot(val_auc, label="Validation AUC")
        plt.xlabel("Boosting Round")
        plt.ylabel("AUC")
        plt.title("Learning Curve (AUC)")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(plot_dir, "training_curve_auc.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] 学習曲線（AUC）を保存しました: {plot_path}")
    else:
        print("[WARN] evals_result に AUC の情報がありません。")

    # ⑩ ROC Curve のプロットと保存
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    roc_plot_path = os.path.join(plot_dir, "roc_curve.png")
    plt.savefig(roc_plot_path)
    plt.close()
    print(f"[INFO] ROC AUC の画像を保存しました: {roc_plot_path}")

    # ⑪ 特徴量寄与率のプロットと保存
    # LightGBM の feature_importance を利用（gain ベース）
    import numpy as np
    importance = model.feature_importance(importance_type='gain')
    feature_names = X_train_balanced.columns
    # DataFrame にまとめて、寄与度順にソート
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values(by='importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance (gain)')
    plt.title('Feature Importance')
    plt.tight_layout()
    feat_imp_path = os.path.join(plot_dir, "feature_importance.png")
    plt.savefig(feat_imp_path)
    plt.close()
    print(f"[INFO] 特徴量寄与率の画像を保存しました: {feat_imp_path}")

    # ⑫ 学習済みモデルの保存（テキスト形式で保存）
    model_filename = "models/lightgbm_meta_model.txt"
    os.makedirs(os.path.dirname(model_filename), exist_ok=True)
    model.save_model(model_filename)
    print(f"[INFO] 学習済みモデルを保存しました: {model_filename}")

    # ※ 評価結果（evals_result）も必要に応じて保存可能
    os.makedirs("models", exist_ok=True)
    joblib.dump(evals_result, "models/evals_result.pkl")
    print("[INFO] evals_result を保存しました。")

if __name__ == '__main__':
    main()
