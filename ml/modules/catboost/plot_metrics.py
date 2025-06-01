import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_metrics(metrics, save_dir="plots"):
    """
    metricsの辞書から、分類器と回帰器の学習曲線および
    特徴量寄与率のグラフを描画して保存する関数。
    特徴量寄与率は上位15件のみ表示します。
    分類問題の場合は混同行列も表示します。

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

    # --- 分類器の混同行列 ---
    conf_matrix = metrics.get('classification', {}).get('confusion_matrix', None)
    if conf_matrix is not None:
        plt.figure(figsize=(8, 6))
        conf_matrix_np = np.array(conf_matrix)
        
        # クラス数を取得
        num_classes = conf_matrix_np.shape[0]
        
        # ヒートマップとして混同行列を表示
        sns.heatmap(conf_matrix_np, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(num_classes),
                   yticklabels=range(num_classes))
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        save_path = os.path.join(save_dir, "classification_confusion_matrix.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved confusion matrix: {save_path}")
    
    # --- 分類器の特徴量寄与率（上位15件のみ） ---
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
        
        # 上位15件に制限（もし15件未満なら全て表示）
        top_k = min(15, len(sorted_idx))
        sorted_idx = sorted_idx[:top_k]
        
        features_sorted = [features[i] for i in sorted_idx]
        importances_sorted = [importances[i] for i in sorted_idx]
        
        plt.figure(figsize=(10,6))
        plt.barh(features_sorted, importances_sorted)
        plt.xlabel("Feature Importance")
        plt.title("Classification Feature Importance (Top 15)")
        plt.gca().invert_yaxis()  # 上に高い重要度が来るようにする
        plt.grid(True)
        save_path = os.path.join(save_dir, "classification_feature_importance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved classification feature importance (top {top_k}): {save_path}")

    # --- 回帰器の特徴量寄与率（上位15件のみ） ---
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
        
        # 上位15件に制限（もし15件未満なら全て表示）
        top_k = min(15, len(sorted_idx))
        sorted_idx = sorted_idx[:top_k]
        
        features_sorted = [features[i] for i in sorted_idx]
        importances_sorted = [importances[i] for i in sorted_idx]
        
        plt.figure(figsize=(10,6))
        plt.barh(features_sorted, importances_sorted)
        plt.xlabel("Feature Importance")
        plt.title("Regression Feature Importance (Top 15)")
        plt.gca().invert_yaxis()
        plt.grid(True)
        save_path = os.path.join(save_dir, "regression_feature_importance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved regression feature importance (top {top_k}): {save_path}")


def export_predictions_to_csv(model, x_test, save_dir="plots/catboost"):
    """
    モデルの予測結果をCSVファイルとして出力する関数。
    マルチタスク学習の場合、最初の2つの予測結果を取り出し、
    adj_returnも計算して追加します。

    Parameters:
        model: CatBoostMultiRegressor のインスタンス
        x_test: テストデータの特徴量
        save_dir: CSVファイルの保存先ディレクトリ
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # モデルから予測結果を取得
    if not hasattr(model, 'regression_predictions') or not hasattr(model, 'regression_ground_truth'):
        print("予測結果が見つかりません。evaluate()メソッドを先に実行してください。")
        return
    
    # 予測結果と実際の値を取得
    predictions = np.array(model.regression_predictions)
    ground_truth = np.array(model.regression_ground_truth)
    
    # DataFrameを作成（インデックスは指定せず）
    df_combined = pd.DataFrame()
    
    # 特徴量を追加（必要に応じて）
    # df_combined = pd.concat([df_combined, x_test.iloc[model.regressors.get_feature_importance()]], axis=1)
    
    # 予測結果の最初の2つを追加
    if len(predictions.shape) > 1 and predictions.shape[1] >= 2:
        df_combined['pred_log_return'] = predictions[:, 0]
        df_combined['pred_volatility'] = predictions[:, 1]
        
        # adj_returnを計算
        epsilon = 1e-8  # ゼロ除算防止用
        df_combined['adj_return'] = df_combined['pred_log_return'] / (df_combined['pred_volatility'] + epsilon)
        
        # 実際の値も追加（あれば）
        if len(ground_truth.shape) > 1 and ground_truth.shape[1] >= 2:
            df_combined['true_log_return'] = ground_truth[:, 0]
            df_combined['true_volatility'] = ground_truth[:, 1]
    elif len(predictions.shape) == 1 or predictions.shape[1] == 1:
        # 1次元の場合
        pred_flat = predictions.flatten() if len(predictions.shape) > 1 else predictions
        df_combined['prediction'] = pred_flat
        
        # 実際の値も追加（あれば）
        if len(ground_truth.shape) == 1 or (len(ground_truth.shape) > 1 and ground_truth.shape[1] == 1):
            true_flat = ground_truth.flatten() if len(ground_truth.shape) > 1 else ground_truth
            df_combined['ground_truth'] = true_flat
    else:
        print("予測結果の形状が想定外です。CSVファイルの出力をスキップします。")
        return
    
    # テストデータのインデックスが利用可能な場合は、フィルタリングされたインデックスを取得して設定
    if hasattr(model, '_prepare_regressor_targets') and isinstance(x_test, pd.DataFrame):
        try:
            # モデルのフィルタリングロジックを使用してインデックスを取得
            x_test_filtered, _ = model._prepare_regressor_targets(x_test, pd.DataFrame(index=x_test.index))
            if len(x_test_filtered) == len(df_combined):
                df_combined.index = x_test_filtered.index
                print(f"フィルタリングされたインデックスを設定しました。行数: {len(df_combined)}")
        except Exception as e:
            print(f"インデックス設定中にエラーが発生しました: {e}")
            # インデックスの設定に失敗した場合は、単純な連番インデックスを使用
            df_combined.index = range(len(df_combined))
    
    # CSVファイルとして保存
    csv_path = os.path.join(save_dir, "training_predictions.csv")
    df_combined.to_csv(csv_path)
    print(f"予測結果をCSVファイルに保存しました: {csv_path} (行数: {len(df_combined)})")
