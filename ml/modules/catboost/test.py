import numpy as np
import pandas as pd
import torch
import logging
import pickle
from pathlib import Path
import os
from rich import print

# CatBoost
from catboost import CatBoostRegressor, CatBoostClassifier, Pool

# Optuna 関連（必要に応じて）
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import Trial

# ユーザー定義の前処理済みデータセット（TimeSeriesDataset）など
from modules.rnn_base.rnn_data_process_multitask import TimeSeriesDataset
from modules.catboost.plot_multitask import plot_final_training_results_multitask

# R2スコア算出用
from sklearn.metrics import r2_score

# ロギング設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)


# ==========================================
# CatBoost を用いた二段階モデリングクラス
# ==========================================
class CatBoostTwoStageModel:
    def __init__(self, **kwargs):
        """
        Parameters:
            train_df       : pd.DataFrame
            val_df         : pd.DataFrame (検証用)
            test_df        : pd.DataFrame (テスト用)
            target_columns : list[str] or str （例：["target_price", "target_vol"]）
        """
        self.train_df = kwargs.get('train_df')
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df')
        target = kwargs.get('target_columns', None)
        if isinstance(target, list):
            self.target_columns = target
        else:
            self.target_columns = [target]

        # デフォルトのハイパーパラメータなど
        self.num_epochs = 1000
        self.window_size = 6
        self.early_stopping_rounds = 30
        self.stride = 90

        # 学習・評価で記録する値
        self.evals_result = {
            'train_loss': [],
            'val_loss': [],
            'test_labels': [],
            'test_preds': [],
            'r2_score': None
        }

        self.best_val_loss = float('inf')
        self.best_hyperparams = {}

        # データ作成（TimeSeriesDataset を利用）
        self.data_set = TimeSeriesDataset(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            target_columns=self.target_columns,
            window_size=self.window_size
        )

    def _prepare_data_for_catboost(self, dataset: TimeSeriesDataset):
        """
        TimeSeriesDataset の (N, window_size, feature_dim) の特徴量と (N, num_target) のラベルを
        (N, window_size * feature_dim) にフラット化して返す。
        """
        X_train = None
        y_train = None
        X_val = None
        y_val = None

        if dataset.train_features is not None:
            train_feat_np = dataset.train_features.numpy()
            N, W, F = train_feat_np.shape
            X_train = train_feat_np.reshape(N, W * F)
            y_train = dataset.train_labels.numpy()

        if dataset.val_features is not None:
            val_feat_np = dataset.val_features.numpy()
            N, W, F = val_feat_np.shape
            X_val = val_feat_np.reshape(N, W * F)
            y_val = dataset.val_labels.numpy()

        return X_train, y_train, X_val, y_val

    def _prepare_test_data_for_catboost(self, dataset: TimeSeriesDataset):
        """
        テスト用データを (N, window_size * feature_dim) にフラット化。
        """
        X_test = None
        y_test = None

        if dataset.test_features is not None:
            test_feat_np = dataset.test_features.numpy()
            N, W, F = test_feat_np.shape
            X_test = test_feat_np.reshape(N, W * F)
            y_test = dataset.test_labels.numpy()

        return X_test, y_test

    def run(self, fold: int = None):
        """
        二段階モデリングのメイン実行。
          ① 分類器（Classifier）でトレード発生の有無を予測
          ② トレードが発生しているサンプルのみを対象に回帰器（Regressor）で各連続値を予測
        """
        # データ準備
        X_train, y_train, X_val, y_val = self._prepare_data_for_catboost(self.data_set)

        # --- ① 分類器の学習 ---
        # 各サンプルについて、いずれかのターゲットが0でなければトレードあり（1）、すべて0ならノートレード（0）とする
        y_train_binary = (y_train != 0).any(axis=1).astype(int)
        y_val_binary = (y_val != 0).any(axis=1).astype(int) if y_val is not None else None

        classifier = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.1,
            loss_function='Logloss',
            early_stopping_rounds=self.early_stopping_rounds,
            random_seed=42,
            verbose=False
        )
        train_pool_clf = Pool(data=X_train, label=y_train_binary)
        val_pool_clf = Pool(data=X_val, label=y_val_binary) if X_val is not None else None
        classifier.fit(train_pool_clf, eval_set=val_pool_clf)

        # --- ② 回帰器の学習 ---
        # トレードが発生している（binary==1）サンプルのみ抽出して回帰モデルを学習
        idx_train_trade = (y_train_binary == 1)
        if np.sum(idx_train_trade) > 0:
            X_train_reg = X_train[idx_train_trade]
            y_train_reg = y_train[idx_train_trade]
        else:
            logger.warning("学習データにトレード発生サンプルが存在しません。全データを使用します。")
            X_train_reg = X_train
            y_train_reg = y_train

        if X_val is not None and y_val_binary is not None:
            idx_val_trade = (y_val_binary == 1)
            if np.sum(idx_val_trade) > 0:
                X_val_reg = X_val[idx_val_trade]
                y_val_reg = y_val[idx_val_trade]
            else:
                X_val_reg = X_val
                y_val_reg = y_val
        else:
            X_val_reg = None
            y_val_reg = None

        regressor = CatBoostRegressor(
            loss_function='MultiRMSE',
            iterations=self.num_epochs,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            bagging_temperature=0.5,
            early_stopping_rounds=self.early_stopping_rounds,
            random_seed=42,
            verbose=False,
            max_bin=254
        )
        train_pool_reg = Pool(data=X_train_reg, label=y_train_reg)
        val_pool_reg = Pool(data=X_val_reg, label=y_val_reg) if X_val_reg is not None else None
        regressor.fit(train_pool_reg, eval_set=val_pool_reg)

        # --- ③ 推論 ---
        X_test, y_test = self._prepare_test_data_for_catboost(self.data_set)
        # ① 分類器で各サンプルのトレード有無を予測
        y_test_binary_pred = classifier.predict(X_test)
        # ② 回帰器の予測値を得る（全サンプルに対して算出）
        y_test_reg_pred = regressor.predict(X_test)
        # shape 調整（1次元の場合は2次元に変換）
        if len(y_test_reg_pred.shape) == 1:
            y_test_reg_pred = y_test_reg_pred.reshape(-1, 1)
        # 分類器でトレードありと予測されたサンプルのみ、回帰予測を採用。その他は 0 とする。
        trade_mask = (y_test_binary_pred.reshape(-1, 1) == 1)
        y_test_final_pred = np.where(trade_mask, y_test_reg_pred, np.zeros_like(y_test_reg_pred))

        # 評価（全ターゲットに対して R2 スコアを算出）
        overall_r2 = r2_score(y_test, y_test_final_pred)
        self.evals_result['r2_score'] = overall_r2
        print(f"[Two-Stage Model] Overall Test R2 Score: {overall_r2:.4f} (fold={fold})")

        # 予測値と正解値の保存
        self.evals_result['test_labels'] = y_test.tolist()
        self.evals_result['test_preds'] = y_test_final_pred.tolist()

        # モデルの保存
        save_dir = "models/catboost_twostage"
        os.makedirs(save_dir, exist_ok=True)
        classifier_model_path = os.path.join(
            save_dir,
            f"catboost_classifier_fold_{fold}.cbm" if fold is not None else "catboost_classifier_final.cbm"
        )
        regressor_model_path = os.path.join(
            save_dir,
            f"catboost_regressor_fold_{fold}.cbm" if fold is not None else "catboost_regressor_final.cbm"
        )
        classifier.save_model(classifier_model_path)
        regressor.save_model(regressor_model_path)
        print(f"Classifierモデルを保存しました: {classifier_model_path}")
        print(f"Regressorモデルを保存しました: {regressor_model_path}")

        return {"r2_score": overall_r2}


# ===============================
# メイン処理例
# ===============================
if __name__ == '__main__':
    from datetime import datetime, timezone, timedelta

    # 例：train_df, val_df, test_df, target_columns は各自定義してください
    train_df = pd.DataFrame()  # 実データの読み込み処理を記述
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    target_columns = ["target_price", "target_vol"]

    model_instance = CatBoostTwoStageModel(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_columns=target_columns
    )

    results = model_instance.run()
    print("最終評価結果:", results)
