import pandas as pd
import lightgbm as lgb
import numpy as np


def predict(df: pd.DataFrame, model_path, meta_model_path=None) -> pd.DataFrame:
    # プライマリモデルのロード
    primary_model = lgb.Booster(model_file=model_path)
    
    # プライマリモデルによる予測確率の取得
    primary_predictions_proba = primary_model.predict(df)
    primary_pred_class = np.argmax(primary_predictions_proba, axis=1)
    
    # プライマリモデルの予測結果をデータフレームに追加
    df['primary_pred_class'] = primary_pred_class
    for i in range(primary_predictions_proba.shape[1]):
        df[f'primary_pred_proba_class_{i}'] = primary_predictions_proba[:, i]
    
    if meta_model_path:
        # メタモデルのロード
        meta_model = lgb.Booster(model_file=meta_model_path)
        # 訓練時と同じ順序で特徴量が揃っていることを前提
        meta_features = df.copy()
        
        # メタモデルによる予測
        meta_predictions = meta_model.predict(meta_features)
        
        # メタモデルがバイナリ分類の場合の処理（閾値0.5）
        meta_pred = np.where(meta_predictions > 0.5, 1, 0)
        
        # 予測結果をデータフレームに追加
        df['meta_predictions'] = meta_pred
    
    # プライマリモデルの予測クラスをデータフレームに追加
    df['predictions'] = primary_pred_class
    
    return df
