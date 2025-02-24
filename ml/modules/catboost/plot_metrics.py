import os
import matplotlib.pyplot as plt

def plot_metrics(metrics, save_dir="plots"):
    """
    metricsの辞書から、分類器と回帰器の学習曲線および
    特徴量寄与率のグラフを描画して保存する関数。

    Parameters:
        metrics (dict): evaluate関数の戻り値の辞書
        save_dir (str): 画像保存先のディレクトリ（存在しなければ自動生成）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # --- 分類器の学習曲線 ---
    clf_lc = metrics.get('classification', {}).get('learning_curve', None)
    if clf_lc is not None:
        # 例: clf_lc = {'learn': {'AUC': [..]}, 'validation': {'AUC': [..]}}
        for metric_name in clf_lc.get('learn', {}):
            plt.figure()
            iterations = range(1, len(clf_lc['learn'][metric_name]) + 1)
            plt.plot(iterations, clf_lc['learn'][metric_name], label='Train')
            if 'validation' in clf_lc and metric_name in clf_lc['validation']:
                plt.plot(iterations, clf_lc['validation'][metric_name], label='Validation')
            plt.title(f'Classification Learning Curve - {metric_name}')
            plt.xlabel('Iterations')
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(save_dir, f"classification_learning_curve_{metric_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved classification learning curve: {save_path}")

    # --- 回帰器の学習曲線 ---
    reg_lc = metrics.get('regression', {}).get('learning_curve', None)
    if reg_lc is not None:
        for metric_name in reg_lc.get('learn', {}):
            plt.figure()
            iterations = range(1, len(reg_lc['learn'][metric_name]) + 1)
            plt.plot(iterations, reg_lc['learn'][metric_name], label='Train')
            if 'validation' in reg_lc and metric_name in reg_lc['validation']:
                plt.plot(iterations, reg_lc['validation'][metric_name], label='Validation')
            plt.title(f'Regression Learning Curve - {metric_name}')
            plt.xlabel('Iterations')
            plt.ylabel(metric_name)
            plt.legend()
            plt.grid(True)
            save_path = os.path.join(save_dir, f"regression_learning_curve_{metric_name}.png")
            plt.savefig(save_path)
            plt.close()
            print(f"Saved regression learning curve: {save_path}")

    # --- 分類器の特徴量寄与率 ---
    clf_fi = metrics.get('classification', {}).get('feature_importance', None)
    if clf_fi is not None:
        if isinstance(clf_fi, dict):
            features = list(clf_fi.keys())
            importances = list(clf_fi.values())
        elif isinstance(clf_fi, list):
            # 特徴量名がない場合はインデックスを利用
            features = list(range(len(clf_fi)))
            importances = clf_fi
        else:
            features, importances = [], []
        
        # 重要度の高い順に並べ替え
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        features_sorted = [features[i] for i in sorted_idx]
        importances_sorted = [importances[i] for i in sorted_idx]
        
        plt.figure(figsize=(10,6))
        plt.barh(features_sorted, importances_sorted)
        plt.xlabel("Feature Importance")
        plt.title("Classification Feature Importance")
        plt.gca().invert_yaxis()  # 上に高い重要度が来るようにする
        plt.grid(True)
        save_path = os.path.join(save_dir, "classification_feature_importance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved classification feature importance: {save_path}")

    # --- 回帰器の特徴量寄与率 ---
    reg_fi = metrics.get('regression', {}).get('feature_importance', None)
    if reg_fi is not None:
        if isinstance(reg_fi, dict):
            features = list(reg_fi.keys())
            importances = list(reg_fi.values())
        elif isinstance(reg_fi, list):
            features = list(range(len(reg_fi)))
            importances = reg_fi
        else:
            features, importances = [], []
        
        sorted_idx = sorted(range(len(importances)), key=lambda i: importances[i], reverse=True)
        features_sorted = [features[i] for i in sorted_idx]
        importances_sorted = [importances[i] for i in sorted_idx]
        
        plt.figure(figsize=(10,6))
        plt.barh(features_sorted, importances_sorted)
        plt.xlabel("Feature Importance")
        plt.title("Regression Feature Importance")
        plt.gca().invert_yaxis()
        plt.grid(True)
        save_path = os.path.join(save_dir, "regression_feature_importance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved regression feature importance: {save_path}")
