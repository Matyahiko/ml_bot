# timesnet_classifier.py

import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
import joblib
import pickle
from sklearn.model_selection import train_test_split
from neuralforecast import NeuralForecast
from neuralforecast.utils import augment_calendar_df
from neuralforecast.models import TimesNet
from neuralforecast.losses.pytorch import DistributionLoss
from torch.utils.data import DataLoader
import torch

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 新しく追加するインポート
from modules.meta.meta_model import train_meta_model
from modules.nn.plot_utils import plot_forecasts

class Config:
    MODEL_DIR = 'models/timesnet/'
    PREDICT_DIR = 'storage/predictions/'

    @staticmethod
    def ensure_directories():
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.PREDICT_DIR, exist_ok=True)  # PREDICT_DIRも作成

class MetaCryptoTimesNet:
    
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None, test_df: pd.DataFrame = None, target_column: str = 'target', device='cuda'):
    
    def __init__(self, train_df, val_df, test_df, target_column,device, window_size, static_df=None, calendar_cols=None):
        self.train_df = train_df
        self.test_df = test_df
        self.window_size = window_size
        self.static_df = static_df
        self.calendar_cols = calendar_cols
        self.model = None
        self.plot_forecasts = plot_forecasts


    def run(self, fold: int = None):
        logger.info("TimesNet分類モデルのトレーニングを開始します。")

        # データの準備
        train_data = self.train_df.copy()
        test_data = self.test_df.copy()

        # unique_idの設定（複数時系列の場合は適宜設定）
        train_data['unique_id'] = train_data.get('unique_id', 0)
        test_data['unique_id'] = test_data.get('unique_id', 0)

        # インデックスのリセット
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        # timestampをdatetimeに変換
        train_data['ds'] = pd.to_datetime(train_data['timestamp'])
        test_data['ds'] = pd.to_datetime(test_data['timestamp'])

        # timestamp列の削除
        train_data.drop(columns=['timestamp'], inplace=True)
        test_data.drop(columns=['timestamp'], inplace=True)

        # target列を'y'にリネーム
        train_data.rename(columns={'target': 'y'}, inplace=True)
        test_data.rename(columns={'target': 'y'}, inplace=True)

        # カレンダー情報の追加が必要な場合
        if self.calendar_cols is not None and self.static_df is not None:
            train_data = augment_calendar_df(df=train_data, freq='D', static_df=self.static_df, calendar_cols=self.calendar_cols)
            test_data = augment_calendar_df(df=test_data, freq='D', static_df=self.static_df, calendar_cols=self.calendar_cols)

        # TimesNetモデルの初期化
        self.model = TimesNet(
            input_size=self.window_size,
            hidden_size=256,
            conv_hidden_size=128,  # 追加パラメータ
            dropout=0.05,
            h=3,
            loss=DistributionLoss(distribution='Normal', level=[80, 90]),  # 追加パラメータ
            futr_exog_list=self.calendar_cols if self.calendar_cols else None,  # 外生変数があれば設定
            scaler_type='standard',
            learning_rate=1e-3,
            max_steps=1000,  # 最大ステップ数を増やす
            val_check_steps=50,
            early_stop_patience_steps=10
        )

        # NeuralForecastの初期化
        nf = NeuralForecast(
            models=[self.model],
            freq='D',  # データの頻度に応じて調整 ('H' for hourly, 'M' for monthly, etc.)
            static_df=self.static_df  # 静的特徴量があれば設定
        )

        # モデルのフィッティング
        logger.info("モデルのフィッティングを開始します。")
        nf.fit(
            df=train_data,
            val_size=12,  # バリデーションサイズを設定
            verbose=True
        )
        logger.info("モデルのフィッティングが完了しました。")

        # モデルの保存
        self.save_model(fold=fold)

        if test_data is not None and not test_data.empty:
            # 予測の実行
            logger.info("予測を実行します。")
            forecasts = nf.predict(futr_df=test_data)

            # 予測結果の整形
            Y_hat_df = forecasts.reset_index(drop=False).drop(columns=['unique_id', 'ds'])
            plot_df = pd.concat([test_data, Y_hat_df], axis=1)
            plot_df = pd.concat([train_data, plot_df], ignore_index=True)

            # 評価
            logger.info("モデルの評価を開始します。")
            evaluation_results = self.evaluate_model(nf, test_data, fold=fold)

            # メタラベリング用データセットの作成
            #meta_data = self.create_meta_dataset(nf, test_data, fold=fold)

            # メタモデルのトレーニングを外部関数に委譲
            #self.train_meta_model_external(meta_data, fold=fold)

            # プロットの実行
            unique_id_to_plot = test_data['unique_id'].unique()[0] if 'unique_id' in test_data.columns else 0
            self.plot_forecasts(plot_df, unique_id=unique_id_to_plot)

            return {
                "fold": fold,
                "evaluation_results": evaluation_results
            }
        else:
            logger.warning("テストデータが提供されていないため、メタモデルのトレーニングをスキップします。")
            return {
                "fold": fold,
                "evaluation_results": None
            }


    def save_model(self, fold: int = None):
        base_name = f'_fold{fold}' if fold is not None else ''
        model_path = os.path.join(Config.MODEL_DIR, f'timesnet_model{base_name}.pth')

        # Save the TimesNet model
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"TimesNetモデルを {model_path} に保存しました。")

    