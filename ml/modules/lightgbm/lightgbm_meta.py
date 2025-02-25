# meta_model_preprocess_training.py

import os
import logging
import pandas as pd
import numpy as np
import pickle

from modules.meta.meta_model import train_meta_model  # メタモデルの学習関数をインポート

logger = logging.getLogger(__name__)

class ConfigMeta:
    MODEL_DIR = 'models/lightgbm/'
    PREDICT_DIR = 'storage/predictions/'

    @staticmethod
    def ensure_directories():
        os.makedirs(ConfigMeta.MODEL_DIR, exist_ok=True)
        os.makedirs(ConfigMeta.PREDICT_DIR, exist_ok=True)

def create_meta_dataset(model, X: pd.DataFrame, y: pd.Series, fold: int = None):
    logger.info("メタモデル用データセットを作成しています。")

    # プライマリモデルの予測確率
    predictions_proba = model.predict(X)
    predictions_class = np.argmax(predictions_proba, axis=1)

    # メタターゲット(正解/不正解)
    meta_labels = (predictions_class == y).astype(int)

    # メタモデル用特徴量
    meta_features = X.copy()
    for i in range(predictions_proba.shape[1]):
        meta_features[f'primary_pred_proba_class_{i}'] = predictions_proba[:, i]
    meta_features['primary_pred_class'] = predictions_class

    meta_df = meta_features.copy()
    meta_df['meta_target'] = meta_labels

    # 必要に応じて保存など
    ConfigMeta.ensure_directories()
    if fold is not None:
        meta_path = os.path.join(ConfigMeta.PREDICT_DIR, f'meta_dataset_fold{fold}.csv')
        meta_df.to_csv(meta_path, index=False)
        logger.info(f"メタモデル用データセットを {meta_path} に保存しました。")

    return meta_df

def train_meta_model_external(meta_df: pd.DataFrame, fold: int = None, device: str = '0'):
    logger.info("外部の関数を使用してメタモデルをトレーニングします。")

    meta_target = meta_df['meta_target'].values

    # プライマリモデルの予測確率カラムだけ抽出
    proba_cols = [col for col in meta_df.columns if 'primary_pred_proba_class_' in col]
    original_feature_cols = [col for col in meta_df.columns if col not in ['meta_target']]

    # 学習用特徴量
    original_features_df = meta_df[original_feature_cols].copy()
    # train_meta_model で要求される形式に合わせてリスト化
    predictions_list = meta_df[proba_cols].values.tolist()

    # メタモデルの保存先
    meta_model_save_path = os.path.join(
        ConfigMeta.MODEL_DIR,
        f'meta_model_fold{fold}' if fold else 'meta_model'
    )
    
    # メタモデルを学習
    meta_model = train_meta_model(
        original_features_df=original_features_df,
        predictions_list=predictions_list,
        meta_target=meta_target,
        model_save_path=meta_model_save_path,
        device=device
    )

    logger.info("メタモデルのトレーニングが完了しました。")
    return meta_model
