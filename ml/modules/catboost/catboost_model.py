import os
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

# ユーザー定義の前処理済みデータセット（TimeSeriesDataset）
from modules.rnn_base.rnn_data_process_multitask import TimeSeriesDataset
from rich import print

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)


class CatBoostMultiRegressor:
    def __init__(self, train_df, test_df, val_df=None, target_columns=None,
                 window_size=1, stride=1, num_epochs=1000, early_stopping_rounds=30,
                 hyperparams=None):
        """
        Parameters:
          - train_df, val_df, test_df : データフレーム
          - target_columns            : 対象カラム（リストまたは単一文字列）
          - window_size, stride       : 時系列切り出し用パラメータ
          - hyperparams               : dict形式のハイパーパラメータ（例: depth, learning_rate など）
        """
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.target_columns = target_columns if isinstance(target_columns, list) else [target_columns]
        #過去データはテクニカル指標で入っているとする   
        #self.window_size = window_size
        self.stride = stride
        self.num_epochs = num_epochs
        self.early_stopping_rounds = early_stopping_rounds

        # ハイパーパラメータ（与えられなければデフォルト値を利用）
        self.depth = hyperparams.get("depth", 6) if hyperparams else 6
        self.learning_rate = hyperparams.get("learning_rate", 0.05) if hyperparams else 0.05
        self.l2_leaf_reg = hyperparams.get("l2_leaf_reg", 3.0) if hyperparams else 3.0
        self.bagging_temperature = hyperparams.get("bagging_temperature", 0.5) if hyperparams else 0.5
        # 以下は過学習対策用のハイパーパラメータ
        self.random_strength = hyperparams.get("random_strength", 1.0) if hyperparams else 1.0
        self.min_data_in_leaf = hyperparams.get("min_data_in_leaf", 1) if hyperparams else 1
        self.rsm = hyperparams.get("rsm", 1.0) if hyperparams else 1.0

        self.best_hyperparams = hyperparams if hyperparams else {}
        self.evals_result = {}

        # TimeSeriesDataset によるデータ作成（本来はフラット化のみ利用）
        self.data_set = TimeSeriesDataset(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            target_columns=self.target_columns,
            window_size=self.window_size,
            stride=self.stride
        )

    def _prepare_data_for_catboost(self, dataset: TimeSeriesDataset):
        X_train, y_train, X_val, y_val = None, None, None, None
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
        X_test, y_test = None, None
        if dataset.test_features is not None:
            test_feat_np = dataset.test_features.numpy()
            N, W, F = test_feat_np.shape
            X_test = test_feat_np.reshape(N, W * F)
            y_test = dataset.test_labels.numpy()
        return X_test, y_test

    def train_and_evaluate(self, fold=None):
        """
        学習・評価・モデル保存の一連の処理。
        """
        X_train, y_train, X_val, y_val = self._prepare_data_for_catboost(self.data_set)
        X_test, y_test = self._prepare_test_data_for_catboost(self.data_set)

        train_pool = Pool(data=X_train, label=y_train) if X_train is not None else None
        val_pool = Pool(data=X_val, label=y_val) if X_val is not None else None

        # GPU設定（ハードコード）
        task_type = 'GPU'
        devices = '0:1'
        cat_model = CatBoostRegressor(
            loss_function='MultiRMSE',
            boosting_type='Plain',
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            bagging_temperature=self.bagging_temperature,
            random_seed=42,
            iterations=self.num_epochs,
            task_type=task_type,
            devices=devices,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
            verbose=False,
            max_bin=254,
            random_strength=self.random_strength,
            min_data_in_leaf=self.min_data_in_leaf,
            rsm=self.rsm
        
        )

        if train_pool is not None:
            cat_model.fit(train_pool, eval_set=val_pool)

        # 学習曲線・特徴量寄与率などの記録
        evals = cat_model.get_evals_result()
        self.evals_result['train_loss'] = evals.get('learn', {}).get('MultiRMSE', [])
        self.evals_result['val_loss'] = evals.get('validation', {}).get('MultiRMSE', [])
        best_score = cat_model.get_best_score()
        val_loss = best_score["validation"]["MultiRMSE"] if val_pool is not None else float('inf')
        self.best_val_loss = val_loss
        self.evals_result['feature_importance'] = cat_model.get_feature_importance(train_pool).tolist()

        # テストデータでの評価
        if X_test is not None:
            test_preds = cat_model.predict(X_test)
            if len(test_preds.shape) == 1:
                test_preds = test_preds.reshape(-1, 1)
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)
            print(f"Best Hyperparameters: {self.best_hyperparams}")
            overall_r2 = r2_score(y_test, test_preds)
            self.evals_result['r2_score'] = overall_r2
            print(f"[CatBoost] Overall Test R2 Score: {overall_r2:.4f} (fold={fold})")

            # ターゲットごとの評価
            self.evals_result['test_labels'] = y_test.tolist()
            self.evals_result['test_preds'] = test_preds.tolist()
            r2_score_per_target = {}
            test_labels_per_target = {}
            test_preds_per_target = {}
            for i, target in enumerate(self.target_columns):
                true_values = y_test[:, i]
                pred_values = test_preds[:, i]
                r2_target = r2_score(true_values, pred_values)
                r2_score_per_target[target] = r2_target
                test_labels_per_target[target] = true_values.tolist()
                test_preds_per_target[target] = pred_values.tolist()
                print(f"[CatBoost] Test R2 Score for {target}: {r2_target:.4f} (fold={fold})")
            self.evals_result['r2_score_per_target'] = r2_score_per_target
            self.evals_result['test_labels_per_target'] = test_labels_per_target
            self.evals_result['test_preds_per_target'] = test_preds_per_target

        # モデル保存処理
        save_dir = "models/catboost"
        os.makedirs(save_dir, exist_ok=True)
        model_filename = f"catboost_model_fold_{fold}.cbm" if fold is not None else "catboost_model_final.cbm"
        save_path = os.path.join(save_dir, model_filename)
        cat_model.save_model(save_path)
        print(f"モデルを保存しました: {save_path}")

        return {"val_loss": val_loss, "r2_score": self.evals_result.get('r2_score')}

    def run(self, fold=None):
        """
        外部からの呼び出し用。run_fold 等からこの run を利用して学習・評価を実施する。
        """
        result_metrics = self.train_and_evaluate(fold=fold)
        return {
            'evaluation_results': {
                'primary_model': {
                    'val_loss': result_metrics.get("val_loss", None),
                    'r2_score': result_metrics.get("r2_score", None)
                }
            }
        }
