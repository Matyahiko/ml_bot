import os 
from rich import print
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score


class CatBoostMultiRegressor:
    def __init__(self, **kwargs):
        
        self.target_columns = kwargs.get("target_columns")
        
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
            'regression': {}
        }
    
    def train(self, x_train, y_train, x_test, y_test):
        # 回帰器の学習
        reg = CatBoostRegressor(**self.regressor_params)
        reg.fit(x_train, y_train, 
                eval_set=(x_test, y_test), 
                use_best_model=True)
        
        self.regressor = reg
    
    def evaluate(self, x_test, y_test):
        metrics = {
            'regression': {}
        }

        # --- 回帰評価 ---
        pred_reg = self.regressor.predict(x_test)
        
        # DataFrameとndarrayの変換処理を適切に行う
        if isinstance(pred_reg, pd.DataFrame):
            self.regression_predictions = pred_reg.values.tolist()
        else:
            self.regression_predictions = pred_reg.tolist() if hasattr(pred_reg, 'tolist') else pred_reg
            
        if isinstance(y_test, pd.DataFrame):
            self.regression_ground_truth = y_test.values.tolist()
            y_true = y_test.values
        else:
            self.regression_ground_truth = y_test.tolist() if hasattr(y_test, 'tolist') else y_test
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
        metrics['regression']['learning_curve'] = self.regressor.get_evals_result()

        # 回帰器の特徴量寄与率を追加
        if isinstance(x_test, pd.DataFrame):
            metrics['regression']['feature_importance'] = dict(zip(x_test.columns, self.regressor.get_feature_importance()))
        else:
            metrics['regression']['feature_importance'] = self.regressor.get_feature_importance().tolist()

        return metrics
    
    def predict(self, x):
        """
        新しいデータに対して予測を行います。
        
        Parameters:
        x: 予測対象の特徴量データ
        
        Returns:
        予測値
        """
        return self.regressor.predict(x)
    
    def save_models(self, save_dir="models/catboost"):
        """
        モデルを保存します。元のクラスとの互換性のため、メソッド名は複数形となっています。
        
        Parameters:
        save_dir: モデルを保存するディレクトリパス
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 回帰器の保存
        reg_path = os.path.join(save_dir, "catboost_regressor.cbm")
        self.regressor.save_model(reg_path)
        print(f"[green]Regressor model saved to {reg_path}[/green]")
    
    def load_model(self, model_path):
        """
        保存済みのモデルをロードします。
        
        Parameters:
        model_path: モデルファイルのパス
        """
        self.regressor = CatBoostRegressor()
        self.regressor.load_model(model_path)
        print(f"[green]Regressor model loaded from {model_path}[/green]")