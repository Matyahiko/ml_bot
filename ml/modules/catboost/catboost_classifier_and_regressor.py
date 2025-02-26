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
            'iterations': 10000,
            'learning_rate': 0.03,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'boosting_type' :'Plain',
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
            'iterations': 5000,
            'learning_rate': 0.003,
            'depth': 7,
            'loss_function': 'MultiRMSE',
            'boosting_type' :'Plain', 
            'task_type': 'GPU',
            'devices': '0:1',  # GPU利用
            'early_stopping_rounds': 100,
            'verbose': 100,
            'random_seed': 42,
            'max_bin': 254,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
        }
        # ユーザー指定の回帰器パラメータで上書き
        user_regressor_params = kwargs.get("regressor_params", {})
        self.regressor_params.update(user_regressor_params)

        # 評価指標の保存用
        self.metrics = {
            'classification': {},
            'regression': {},
            'overall': {}
        }
    
    def _prepare_binary_targets(self, x, y, downsample=True):
        # xとyを結合してから判定する
        df = pd.concat([x, y], axis=1)
        features_df = df[x.columns]
        target_df = pd.DataFrame()
        target_df["target"] = (df[self.target_columns] != 0).any(axis=1).astype(int)
        
        if not downsample:
            return features_df, target_df
            
        # ダウンサンプリングの実装
        positive_samples = features_df[target_df["target"] == 1]
        negative_samples = features_df[target_df["target"] == 0]
        positive_targets = target_df[target_df["target"] == 1]
        negative_targets = target_df[target_df["target"] == 0]
        
        # 少数派クラスのサンプル数に合わせる
        pos_count = len(positive_samples)
        if pos_count < len(negative_samples):
            # 多数派（0）をダウンサンプリング
            neg_downsampled_idx = np.random.choice(
                negative_samples.index, pos_count, replace=False
            )
            negative_samples = negative_samples.loc[neg_downsampled_idx]
            negative_targets = negative_targets.loc[neg_downsampled_idx]
        
        # バランスの取れたデータセットを作成
        balanced_features = pd.concat([positive_samples, negative_samples])
        balanced_targets = pd.concat([positive_targets, negative_targets])
        
        # ランダムにシャッフル
        shuffle_idx = np.random.permutation(len(balanced_features))
        balanced_features = balanced_features.iloc[shuffle_idx].reset_index(drop=True)
        balanced_targets = balanced_targets.iloc[shuffle_idx].reset_index(drop=True)
        
        return balanced_features, balanced_targets

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
        # 二値分類用ターゲットの作成（ダウンサンプリング適用）
        x_train_balanced, train_binary_targets = self._prepare_binary_targets(x_train, y_train, downsample=True)
        x_test_balanced, test_binary_targets = self._prepare_binary_targets(x_test, y_test, downsample=True)
        
        # 分類器の学習
        clf = CatBoostClassifier(**self.classifier_params)
        clf.fit(x_train_balanced, train_binary_targets, 
                eval_set=[(x_test_balanced, test_binary_targets)],  # リストに包む
                use_best_model=True)
        
        # 回帰器用にデータを準備
        x_train_reg, y_train_reg = self._prepare_regressor_targets(x_train, y_train)
        x_test_reg, y_test_reg = self._prepare_regressor_targets(x_test, y_test)
        
        # 回帰器の学習 - eval_setを正しく設定
        reg = CatBoostRegressor(**self.regressor_params)
        reg.fit(x_train_reg, y_train_reg, 
                eval_set=[(x_test_reg, y_test_reg)],  # リストに包む
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
        self.regression_ground_truth = y_test.values.tolist()
        y_true = y_test.values
        
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

        self.metrics = metrics
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
        
    def print_results(self, detailed=False):
        """
        評価結果を出力します。
        
        Parameters:
        detailed: True の場合、詳細な結果を出力します
        """
        if not hasattr(self, 'metrics') or not self.metrics:
            print("[yellow]評価メトリクスがありません。evaluate()メソッドを先に実行してください。[/yellow]")
            return
        
        # 分類モデルの評価結果を出力
        print("\n[bold blue]===== 分類モデル評価結果 =====[/bold blue]")
        
        clf_metrics = self.metrics['classification']
        
        # 学習パラメータの表示
        print("\n[bold cyan]学習パラメータ:[/bold cyan]")
        for param, value in self.classifier_params.items():
            print(f"  {param}: {value}")
        
        # 分類評価指標の表示
        print(f"\n[bold green]評価指標:[/bold green]")
        print(f"  精度 (Accuracy): {clf_metrics['accuracy']:.4f}")
        print(f"  適合率 (Precision): {clf_metrics['precision']:.4f}")
        print(f"  再現率 (Recall): {clf_metrics['recall']:.4f}")
        print(f"  F1スコア: {clf_metrics['f1']:.4f}")
        
        # 詳細モードの場合は特徴量重要度も表示
        if detailed and 'feature_importance' in clf_metrics:
            print("\n[bold magenta]特徴量重要度 (上位10件):[/bold magenta]")
            
            # 特徴量重要度を取得してソート
            feature_imp = clf_metrics['feature_importance']
            
            if isinstance(feature_imp, dict):
                # 重要度の高い順にソート
                sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
                
                # 上位10件を表示
                for feature, importance in sorted_features[:10]:
                    print(f"  {feature}: {importance:.4f}")
            else:
                print(f"  特徴量重要度: {feature_imp}")
        
        # 学習曲線の情報を表示
        if detailed and 'learning_curve' in clf_metrics:
            learn_curve = clf_metrics['learning_curve']
            print("\n[bold yellow]学習曲線情報:[/bold yellow]")
            
            if 'learn' in learn_curve and 'validation' in learn_curve:
                # 最適イテレーション数
                best_iter = self.classifiers.get_best_iteration()
                print(f"  最適イテレーション数: {best_iter}")
                
                # 学習データとバリデーションデータの最終スコア
                learn_metrics = learn_curve['learn']
                valid_metrics = learn_curve['validation']
                
                metric_names = list(learn_metrics.keys())
                for metric in metric_names:
                    final_learn = learn_metrics[metric][-1]
                    final_valid = valid_metrics[metric][-1]
                    print(f"  最終{metric}: 学習={final_learn:.4f}, 検証={final_valid:.4f}")
        
        # 回帰モデルの評価結果を出力（既存のコード）
        print("\n[bold blue]===== 回帰モデル評価結果 =====[/bold blue]")
        
        reg_metrics = self.metrics['regression']
        
        # 学習パラメータの表示
        print("\n[bold cyan]学習パラメータ:[/bold cyan]")
        for param, value in self.regressor_params.items():
            print(f"  {param}: {value}")
        
        # 単一ターゲットか複数ターゲットかで表示方法を分ける
        if 'rmse' in reg_metrics:
            # 単一ターゲットの場合
            rmse = reg_metrics['rmse']
            r2 = reg_metrics['r2']
            print(f"\n[bold green]全体評価指標:[/bold green]")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  R²: {r2:.4f}")
        else:
            # 複数ターゲットの場合
            print(f"\n[bold green]各ターゲット評価指標:[/bold green]")
            
            # ターゲット列名のリストを取得
            target_cols = [col for col in reg_metrics.keys() 
                        if col not in ['learning_curve', 'feature_importance']]
            
            # 各ターゲットの評価指標を表示
            for col in target_cols:
                print(f"\n[bold]ターゲット: {col}[/bold]")
                print(f"  RMSE: {reg_metrics[col]['rmse']:.4f}")
                print(f"  R²: {reg_metrics[col]['r2']:.4f}")
        
        # 詳細モードの場合は特徴量重要度も表示
        if detailed and 'feature_importance' in reg_metrics:
            print("\n[bold magenta]特徴量重要度 (上位10件):[/bold magenta]")
            
            # 特徴量重要度を取得してソート
            feature_imp = reg_metrics['feature_importance']
            
            if isinstance(feature_imp, dict):
                # 重要度の高い順にソート
                sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
                
                # 上位10件を表示
                for feature, importance in sorted_features[:10]:
                    print(f"  {feature}: {importance:.4f}")
            else:
                print(f"  特徴量重要度: {feature_imp}")
        
        # 学習曲線の情報を表示
        if detailed and 'learning_curve' in reg_metrics:
            learn_curve = reg_metrics['learning_curve']
            print("\n[bold yellow]学習曲線情報:[/bold yellow]")
            
            if 'learn' in learn_curve and 'validation' in learn_curve:
                # 最適イテレーション数（self.regressor → self.regressors に修正）
                best_iter = self.regressors.get_best_iteration()
                print(f"  最適イテレーション数: {best_iter}")
                
                # 学習データとバリデーションデータの最終スコア
                learn_metrics = learn_curve['learn']
                valid_metrics = learn_curve['validation']
                
                metric_names = list(learn_metrics.keys())
                for metric in metric_names:
                    final_learn = learn_metrics[metric][-1]
                    final_valid = valid_metrics[metric][-1]
                    print(f"  最終{metric}: 学習={final_learn:.4f}, 検証={final_valid:.4f}")
        
        # 総合評価の表示
        if 'overall' in self.metrics:
            print("\n[bold blue]===== 総合評価 =====[/bold blue]")
            overall_metrics = self.metrics['overall']
            
            if 'geometric_mean' in overall_metrics:
                print(f"  幾何平均スコア: {overall_metrics['geometric_mean']:.4f}")