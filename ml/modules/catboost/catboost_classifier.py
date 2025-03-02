import os 
from rich import print
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class CatBoostMultiClassifier:
    def __init__(self, **kwargs):
        
        self.target_columns = kwargs.get("target_columns")
        
        # デフォルトの分類器パラメータ
        self.classifier_params = {
            'iterations': 10000,
            'learning_rate': 0.03,
            'depth': 7,
            'loss_function': 'MultiClass',  # 多クラス分類用
            'eval_metric': 'MultiClass',
            'boosting_type': 'Plain',
            'task_type': 'GPU',
            'devices': '0:1',  # GPU利用
            'early_stopping_rounds': 30,
            'verbose': 500,
            'random_seed': 42,
            'max_bin': 254,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'l2_leaf_reg': 3,
        }
        # ユーザー指定の分類器パラメータで上書き
        user_classifier_params = kwargs.get("classifier_params", {})
        self.classifier_params.update(user_classifier_params)

        # 評価指標の保存用
        self.metrics = {
            'classification': {}
        }
        
        # クラス数を保存
        self.num_classes = kwargs.get("num_classes", None)
        
    def _determine_num_classes(self, y):
        """データセットからクラス数を推定する"""
        if isinstance(y, pd.DataFrame):
            # 単一ターゲット列の場合
            if len(self.target_columns) == 1:
                return len(y[self.target_columns[0]].unique())
            # 複数ターゲット列の場合（最初の列を使用）
            else:
                return len(y[self.target_columns[0]].unique())
        else:
            # numpy配列の場合
            return len(np.unique(y))
    
    def train(self, x_train, y_train, x_test, y_test):
        # ターゲットデータの準備
        if isinstance(y_train, pd.DataFrame):
            if len(self.target_columns) == 1:
                train_targets = y_train[self.target_columns[0]]
                test_targets = y_test[self.target_columns[0]]
            else:
                # 複数ターゲット列がある場合は最初の列を使用
                train_targets = y_train[self.target_columns[0]]
                test_targets = y_test[self.target_columns[0]]
        else:
            train_targets = y_train
            test_targets = y_test
        
        # クラス数の推定（未設定の場合）
        if self.num_classes is None:
            self.num_classes = self._determine_num_classes(y_train)
            print(f"[green]Detected {self.num_classes} classes from the dataset[/green]")
            
            # クラス数に基づいてパラメータを更新
            if self.num_classes == 2:
                self.classifier_params['loss_function'] = 'Logloss'
                self.classifier_params['eval_metric'] = 'AUC'
            else:
                self.classifier_params['loss_function'] = 'MultiClass'
                self.classifier_params['eval_metric'] = 'MultiClass'
                
            print(f"[green]Using loss function: {self.classifier_params['loss_function']}[/green]")
        
        # 分類器の学習
        clf = CatBoostClassifier(**self.classifier_params)
        clf.fit(x_train, train_targets, 
                eval_set=(x_test, test_targets), 
                use_best_model=True)
        self.classifier = clf
    
    def evaluate(self, x_test, y_test):
        metrics = {
            'classification': {}
        }

        # ターゲットデータの準備
        if isinstance(y_test, pd.DataFrame):
            if len(self.target_columns) == 1:
                test_targets = y_test[self.target_columns[0]]
            else:
                # 複数ターゲット列がある場合は最初の列を使用
                test_targets = y_test[self.target_columns[0]]
        else:
            test_targets = y_test

        # 予測
        pred_proba = self.classifier.predict_proba(x_test)
        pred_class = self.classifier.predict(x_test)
        
        # 評価指標の計算
        metrics['classification']['accuracy'] = accuracy_score(test_targets, pred_class)
        
        # 二値分類の場合
        if self.num_classes == 2:
            metrics['classification']['precision'] = precision_score(test_targets, pred_class)
            metrics['classification']['recall'] = recall_score(test_targets, pred_class)
            metrics['classification']['f1'] = f1_score(test_targets, pred_class)
        else:
            # 多クラス分類の場合
            metrics['classification']['precision'] = precision_score(test_targets, pred_class, average='weighted')
            metrics['classification']['recall'] = recall_score(test_targets, pred_class, average='weighted')
            metrics['classification']['f1'] = f1_score(test_targets, pred_class, average='weighted')
        
        # 混同行列
        metrics['classification']['confusion_matrix'] = confusion_matrix(test_targets, pred_class).tolist()
        
        # 学習曲線
        metrics['classification']['learning_curve'] = self.classifier.get_evals_result()

        # 特徴量重要度
        if isinstance(x_test, pd.DataFrame):
            metrics['classification']['feature_importance'] = dict(
                zip(x_test.columns, self.classifier.get_feature_importance())
            )
        else:
            metrics['classification']['feature_importance'] = self.classifier.get_feature_importance().tolist()

        self.metrics = metrics
        return metrics

    def save_models(self, save_dir="models/catboost"):
        os.makedirs(save_dir, exist_ok=True)
        
        # 分類器の保存
        clf_path = os.path.join(save_dir, "catboost_classifier.cbm")
        self.classifier.save_model(clf_path)
        print(f"[green]Classifier model saved to {clf_path}[/green]")
        
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
        
        # 混同行列の表示
        if 'confusion_matrix' in clf_metrics:
            print("\n[bold yellow]混同行列:[/bold yellow]")
            conf_matrix = np.array(clf_metrics['confusion_matrix'])
            for row in conf_matrix:
                print(f"  {row}")
        
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
                best_iter = self.classifier.get_best_iteration()
                print(f"  最適イテレーション数: {best_iter}")
                
                # 学習データとバリデーションデータの最終スコア
                learn_metrics = learn_curve['learn']
                valid_metrics = learn_curve['validation']
                
                metric_names = list(learn_metrics.keys())
                for metric in metric_names:
                    final_learn = learn_metrics[metric][-1]
                    final_valid = valid_metrics[metric][-1]
                    print(f"  最終{metric}: 学習={final_learn:.4f}, 検証={final_valid:.4f}")
