import os
import logging
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

logger = logging.getLogger(__name__)

class CryptoLightGBM:
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
        device: str,
        val_df: pd.DataFrame = None  # 追加：外部から検証データを渡す場合
    ):
        self.train_df = train_df
        self.test_df = test_df
        self.val_df = val_df  # 検証データがある場合は使用する
        self.target_column = target_column
        self.device = device
        self.evaluation_results = {}
        self.evals_result = {}  # コールバック用の評価結果保持用辞書

        # GPUデバイスIDの設定（旧コードと同様の処理）
        if isinstance(device, str) and device.lower().startswith("cuda:"):
            try:
                self.gpu_device_id = int(device.split(":")[1])
            except Exception as e:
                logger.warning(f"GPUデバイスIDの解析に失敗しました。デフォルトの0を使用します。: {e}")
                self.gpu_device_id = 0
        else:
            self.gpu_device_id = 0

    def run(self, fold: int = None):
        logger.info("LightGBM分類モデルのトレーニングを開始します。")

        # 学習データからクラス数を算出（最新コードではself.num_classesを利用）
        self.num_classes = self.train_df[self.target_column].nunique()

        # 学習用データの準備（特徴量・目的変数の分離）
        X = self.train_df.drop(columns=[self.target_column])
        y = self.train_df[self.target_column]

        # 検証データの用意：事前にval_dfが与えられていればそれを利用、なければtrain_test_splitで分割
        if self.val_df is not None:
            X_train, y_train = X, y
            X_val = self.val_df.drop(columns=[self.target_column])
            y_val = self.val_df[self.target_column]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.1, shuffle=True, stratify=y
            )

        # テストデータの用意（存在する場合）
        if self.test_df is not None:
            X_test = self.test_df.drop(columns=[self.target_column])
            y_test = self.test_df[self.target_column]
        else:
            X_test, y_test = None, None

        # LightGBM用データセットの作成
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

        # パラメータ設定の更新（最新コードを参考）
        params = {
            'objective': 'multiclass',
            'num_class': self.num_classes,
            'metric': ['multi_logloss', 'multi_error'],
            'boosting_type': 'gbdt',
            'learning_rate': 0.01,
            'extra_trees': True,
            'num_leaves': 31,
            'lambda_l2': 1.0,
            'bagging_fraction': 0.9,
            'verbose': -1,
            'device': 'GPU',              # 常にGPUモードで実行
            'gpu_platform_id': 0,         # 追加：GPUプラットフォームID
            'gpu_device_id': self.gpu_device_id  # __init__で設定したIDを使用
        }

        # コールバックの追加（早期終了、ログ出力など）
        callbacks = [
            lgb.record_evaluation(self.evals_result),
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=100)
        ]

        # 学習実行。最新コードではself.modelとして属性に格納
        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=2000,
            valid_sets=[lgb_train, lgb_val],
            valid_names=['train', 'validation'],
            callbacks=callbacks
        )

        # 特徴量名の取得（特徴量重要度プロット用）
        feature_names = X_train.columns.tolist()

        # 学習過程や特徴量重要度のプロット、モデル保存（旧コードと同様の処理）
        self._plot_learning_curve(self.evals_result, fold)
        self._plot_feature_importance(self.model, feature_names, fold)
        self._save_model(self.model, "models/lightgbm", fold)

        evaluation_results = {}
        # テストデータがある場合は評価処理を実施
        if X_test is not None and y_test is not None:
            logger.info("テストデータに対する予測と評価を開始")
            y_prob = self.model.predict(X_test, num_iteration=self.model.best_iteration)
            y_pred = np.argmax(y_prob, axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            multi_logloss = log_loss(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"テストデータ評価結果 - Accuracy: {accuracy:.4f}, Multi-logloss: {multi_logloss:.4f}")
            logger.info(f"混同行列:\n{cm}")
            self._plot_confusion_matrix(cm, sorted(y_test.unique()), fold)
            self._save_predictions(self.test_df, y_pred, "storage/predictions", fold)
            evaluation_results = {
                'accuracy': accuracy,
                'multi_logloss': multi_logloss,
                'confusion_matrix': cm.tolist()
            }

        logger.info("LightGBMトレーニングプロセス完了")
        return {
            'model': self.model,
            'evaluation_results': {
                'primary_model': evaluation_results
            }
        }

    def _preprocess_labels(self, df: pd.DataFrame, target_col: str):
        """
        ターゲットカラム内の値「2」を「1」に置換し、ユニークなラベル数を算出します。
        """
        logger.info("前処理開始: ターゲットカラムの値2を1に置換")
        df[target_col] = df[target_col].replace(2, 1)
        unique_labels = df[target_col].unique()
        num_classes = len(unique_labels)
        logger.info(f"前処理完了: ユニークなラベル数 = {num_classes}")
        return df, num_classes

    def _split_train_validation(self, df: pd.DataFrame, target_col: str, test_size=0.1, random_state=42):
        """
        トレーニングデータから層化サンプリングによりバリデーションデータを抽出します。
        """
        logger.info("トレーニングデータからバリデーションデータを層化サンプリングにより抽出")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        logger.info(f"分割完了: トレーニングデータサイズ = {X_train.shape}, バリデーションデータサイズ = {X_val.shape}")
        return X_train, y_train, X_val, y_val

    def _plot_learning_curve(self, evals_result: dict, fold=None):
        """
        学習曲線（multi_logloss, multi_error）の推移をプロットし、ファイルに保存します。
        """
        logger.info("学習曲線のプロット作成")
        metrics = ['multi_logloss', 'multi_error']
        for metric in metrics:
            if metric in evals_result['training']:
                plt.figure()
                plt.plot(evals_result['training'][metric], label='train')
                if 'valid_1' in evals_result and metric in evals_result['valid_1']:
                    plt.plot(evals_result['valid_1'][metric], label='validation')
                plt.xlabel('Iteration')
                plt.ylabel(metric)
                plt.title(f'Learning Curve ({metric})')
                plt.legend()
                filename = f"plots/lightgbm/learning_curve_{metric}"
                if fold is not None:
                    filename += f"_fold{fold}"
                filename += ".png"
                plt.savefig(filename)
                plt.close()
                logger.info(f"学習曲線のプロット保存: {filename}")

    def _plot_feature_importance(self, model, feature_names: list, fold=None):
        """
        モデルから算出された特徴量重要度をプロットし、ファイルに保存します。
        """
        logger.info("特徴量重要度のプロット作成")
        importance = model.feature_importance()
        importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
        importance_df.sort_values(by='importance', ascending=False, inplace=True)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Feature Importance')
        filename = "plots/lightgbm/feature_importance"
        if fold is not None:
            filename += f"_fold{fold}"
        filename += ".png"
        plt.savefig(filename)
        plt.close()
        logger.info(f"特徴量重要度プロット保存: {filename}")

    def _plot_confusion_matrix(self, cm, class_labels, fold=None):
        """
        混同行列をヒートマップとしてプロットし、ファイルに保存します。
        """
        logger.info("混同行列のプロット作成")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        filename = "plots/lightgbm/confusion_matrix"
        if fold is not None:
            filename += f"_fold{fold}"
        filename += ".png"
        plt.savefig(filename)
        plt.close()
        logger.info(f"混同行列プロット保存: {filename}")

    def _save_model(self, model, model_dir: str, fold=None):
        """
        学習済みモデルをLightGBMネイティブ形式（.txt）およびjoblib形式（.joblib）で保存します。
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"モデル保存ディレクトリ作成: {model_dir}")
        filename_suffix = f"_fold{fold}" if fold is not None else ""
        # LightGBMネイティブ形式で保存
        model_txt_path = os.path.join(model_dir, f"lightgbm_model{filename_suffix}.txt")
        model.save_model(model_txt_path)
        logger.info(f"LightGBMモデル保存 (ネイティブ形式): {model_txt_path}")
        # joblib形式で保存
        model_joblib_path = os.path.join(model_dir, f"lightgbm_model{filename_suffix}.joblib")
        joblib.dump(model, model_joblib_path)
        logger.info(f"LightGBMモデル保存 (joblib形式): {model_joblib_path}")

    def _save_predictions(self, test_df: pd.DataFrame, predictions, output_dir: str, fold=None):
        """
        テストデータに予測結果（predictionカラム）を追加し、CSV形式で保存します。
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"予測結果保存ディレクトリ作成: {output_dir}")
        test_df['prediction'] = predictions
        filename_suffix = f"_fold{fold}" if fold is not None else ""
        output_path = os.path.join(output_dir, f"test_predictions{filename_suffix}.csv")
        test_df.to_csv(output_path, index=False)
        logger.info(f"予測結果CSV保存: {output_path}")
