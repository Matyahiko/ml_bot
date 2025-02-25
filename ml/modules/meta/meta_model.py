# meta_model.py

import os
import logging
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import pickle

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_meta_model(original_features_df: pd.DataFrame, 
                    predictions_list: list, 
                    meta_target: pd.Series,
                    model_save_path: str, 
                    device: str = 'cuda:0') -> lgb.Booster:
    """
    メタモデルをトレーニングし、指定されたパスに保存します。

    Parameters:
    - original_features_df (pd.DataFrame): 元の特徴量データフレーム。
    - predictions_list (list): プライマリモデルの予測確率のリスト。
    - meta_target (pd.Series): メタラベル（プライマリモデルが正解したかどうか）。
    - model_save_path (str): メタモデルを保存するディレクトリのパス。
    - device (str): 使用するデバイス（デフォルトは 'cuda:0'）。

    Returns:
    - meta_model (lgb.Booster): トレーニングされたメタモデル。
    """

    # ディレクトリの作成
    os.makedirs(model_save_path, exist_ok=True)
    logger.info(f"メタモデルを保存するディレクトリ: {model_save_path}")

    # メタモデル用特徴量の作成
    logger.info("メタモデル用の特徴量を作成しています。")
    meta_features = original_features_df.copy()
    predictions_proba = pd.DataFrame(predictions_list, columns=[f'primary_pred_proba_class_{i}' for i in range(len(predictions_list[0]))])
    meta_features = pd.concat([meta_features, predictions_proba], axis=1)
    meta_features['primary_pred_class'] = predictions_proba.idxmax(axis=1).str.replace('primary_pred_proba_class_', '').astype(int)
    meta_features['meta_target'] = meta_target

    # メタデータセットの分割
    X_meta = meta_features.drop(columns=['meta_target'])
    y_meta = meta_features['meta_target']

    X_train_meta, X_val_meta, y_train_meta, y_val_meta = train_test_split(
        X_meta, y_meta, test_size=0.1, shuffle=True, stratify=y_meta, random_state=42
    )

    lgb_train_meta = lgb.Dataset(X_train_meta, y_train_meta)
    lgb_val_meta = lgb.Dataset(X_val_meta, y_val_meta, reference=lgb_train_meta)

    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'learning_rate': 0.01,
        'extra_trees': True,
        'num_leaves': 31,
        'lambda_l2': 1.0,
        'bagging_fraction': 0.9,
        'verbose': -1,
        'device': device,
    }

    evals_result = {}

    callbacks = [
        lgb.record_evaluation(evals_result),
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]

    logger.info("メタモデルのトレーニングを開始します。")
    meta_model = lgb.train(
        params,
        lgb_train_meta,
        num_boost_round=4000,
        valid_sets=[lgb_train_meta, lgb_val_meta],
        valid_names=['train_meta', 'validation_meta'],
        callbacks=callbacks
    )
    logger.info("メタモデルのトレーニングが完了しました。")

    # モデルの保存
    meta_model_path = os.path.join(model_save_path, 'lightgbm_meta_model.txt')
    joblib_path = os.path.join(model_save_path, 'lightgbm_meta_model.joblib')

    meta_model.save_model(meta_model_path)
    logger.info(f"メタモデルを {meta_model_path} に保存しました。")

    with open(joblib_path, 'wb') as f:
        pickle.dump(meta_model, f)
    logger.info(f"メタモデルを {joblib_path} にJoblib形式で保存しました。")

    # メタモデルの評価
    meta_pred_proba = meta_model.predict(X_val_meta, num_iteration=meta_model.best_iteration)
    meta_pred = (meta_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_val_meta, meta_pred)
    logger.info(f"メタモデルの精度: {accuracy:.4f}")

    # 評価結果の保存
    eval_path = os.path.join(model_save_path, 'meta_model_evaluation.txt')
    with open(eval_path, 'w') as f:
        f.write(f"Meta Model Accuracy: {accuracy}\n")
    logger.info(f"メタモデルの評価結果を {eval_path} に保存しました。")

    return meta_model
