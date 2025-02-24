import os 
from rich import print
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score


class CatBoostClassifierAndMultiRegressor:
    def __init__(self, **kwargs):
        
        self.target_columns = target_columns = kwargs.get("target_columns")
        
        # デフォルトの分類器パラメータ
        self.classifier_params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'task_type': 'GPU',
            'devices': '0:1',  # GPU利用
            'early_stopping_rounds': 30,
            'verbose': 100,
            'random_seed': 42
        }
        # ユーザー指定の分類器パラメータで上書き
        user_classifier_params = kwargs.get("classifier_params", {})
        self.classifier_params.update(user_classifier_params)
        
        # デフォルトの回帰器パラメータ
        self.regressor_params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'MultiRMSE',
            'task_type': 'GPU',
            'devices': '0:1',  # GPU利用
            'early_stopping_rounds': 30,
            'verbose': 100,
            'random_seed': 42
        }
        # ユーザー指定の回帰器パラメータで上書き
        user_regressor_params = kwargs.get("regressor_params", {})
        self.regressor_params.update(user_regressor_params)

        # 評価指標の保存用
        self.metrics = {
            'classification': {},
            'regression': {}
        }
    
    def _prepare_binary_targets(self, df):
        target_df = pd.DataFrame()
        target_df["target"] = (df[self.target_columns] != 0).any(axis=1).astype(int)
        return target_df
    
    def _prepare_regressor_targets(self, x, y):
        # xとyを結合
        df = pd.concat([x, y], axis=1)
        # target列のいずれかの値が0でない行だけ抽出
        df_filtered = df[(df[self.target_columns] != 0).any(axis=1)]
        # 結合前の列をそれぞれに戻す
        x_filtered = df_filtered[x.columns]
        y_filtered = df_filtered[y.columns]
        return x_filtered, y_filtered
    
    def train(self, x_train, y_train, x_test, y_test):
        # 二値分類用ターゲットの作成
        self.train_binary_targets = self._prepare_binary_targets(x_train)
        self.test_binary_targets = self._prepare_binary_targets(x_test)
        
        # 取引が起こるかの分類器の学習
        clf = CatBoostClassifier(**self.classifier_params)
        clf.fit(x_train, self.train_binary_targets, 
                eval_set=(x_test, self.test_binary_targets), 
                use_best_model=True)
        
        # 回帰器用に、x_trainとy_trainを結合後、target列が0でない行のみでデータを準備
        x_reg, y_reg = self._prepare_regressor_targets(x_train, y_train)
        
        # 回帰器の学習
        reg = CatBoostRegressor(**self.regressor_params)
        reg.fit(x_reg, y_reg, 
                eval_set=(x_test, y_test), 
                use_best_model=True)
        
        self.classifiers = clf
        self.regressors = reg
    
    def evaluate(self, x_test, y_test):
        metrics = {
            'classification': {},
            'regression': {},
            'overall': {}
        }

        # --- 分類評価 ---
        pred_clf = self.classifiers.predict(x_test)
        true_labels = self.test_binary_targets['target']
        metrics['classification']['accuracy'] = accuracy_score(true_labels, pred_clf)
        metrics['classification']['precision'] = precision_score(true_labels, pred_clf)
        metrics['classification']['recall'] = recall_score(true_labels, pred_clf)
        metrics['classification']['f1'] = f1_score(true_labels, pred_clf)

        # 分類器の学習曲線（学習時の指標推移）を追加
        metrics['classification']['learning_curve'] = self.classifiers.get_evals_result()

        # 分類器の特徴量寄与率を追加（DataFrameの場合、列名付きのdictとして出力）
        if isinstance(x_test, pd.DataFrame):
            metrics['classification']['feature_importance'] = dict(zip(x_test.columns, self.classifiers.get_feature_importance()))
        else:
            metrics['classification']['feature_importance'] = self.classifiers.get_feature_importance().tolist()

        # --- 回帰評価 ---
        pred_reg = self.regressors.predict(x_test)
        self.regression_predictions = pred_reg.tolist()
        self.regression_ground_truth = y_test.tolist()
        y_true = np.array(y_test)
        if y_true.ndim == 1:
            rmse = np.sqrt(np.mean((pred_reg - y_true) ** 2))
            r2 = r2_score(y_true, pred_reg)
            metrics['regression'] = {'rmse': rmse, 'r2': r2}
        elif y_true.ndim == 2:
            metrics['regression'] = {}
            if hasattr(y_test, 'columns'):
                for i, col in enumerate(y_test.columns):
                    rmse = np.sqrt(np.mean((pred_reg[:, i] - y_true[:, i]) ** 2))
                    r2 = r2_score(y_true[:, i], pred_reg[:, i])
                    metrics['regression'][col] = {'rmse': rmse, 'r2': r2}
            else:
                for i in range(y_true.shape[1]):
                    rmse = np.sqrt(np.mean((pred_reg[:, i] - y_true[:, i]) ** 2))
                    r2 = r2_score(y_true[:, i], pred_reg[:, i])
                    metrics['regression'][f"target_{i}"] = {'rmse': rmse, 'r2': r2}

        # 回帰器の学習曲線を追加
        metrics['regression']['learning_curve'] = self.regressors.get_evals_result()

        # 回帰器の特徴量寄与率を追加
        if isinstance(x_test, pd.DataFrame):
            metrics['regression']['feature_importance'] = dict(zip(x_test.columns, self.regressors.get_feature_importance()))
        else:
            metrics['regression']['feature_importance'] = self.regressors.get_feature_importance().tolist()

        # --- 全体評価（例として分類F1スコアと回帰R2スコアの幾何平均） ---
        clf_score = metrics['classification']['f1']
        if isinstance(metrics['regression'], dict) and 'r2' not in metrics['regression']:
            reg_scores = [v['r2'] for k, v in metrics['regression'].items() if isinstance(v, dict) and 'r2' in v]
            reg_score = np.mean(reg_scores) if reg_scores else 0
        else:
            reg_score = metrics['regression'].get('r2', 0)

        overall_score = np.sqrt(clf_score * reg_score) if clf_score and reg_score else 0
        metrics['overall']['geometric_mean'] = overall_score

        return metrics
    
    def save_models(self, save_dir="models/catboost"):
        os.makedirs(save_dir, exist_ok=True)
        
        # 分類器の保存
        clf_path = os.path.join(save_dir, "catboost_classifier.cbm")
        self.classifiers.save_model(clf_path)
        print(f"[green]Classifier model saved to {clf_path}[/green]")
        
        # 回帰器の保存
        reg_path = os.path.join(save_dir, "catboost_regressor.cbm")
        self.regressors.save_model(reg_path)
        print(f"[green]Regressor model saved to {reg_path}[/green]")
