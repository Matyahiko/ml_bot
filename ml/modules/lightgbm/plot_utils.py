# modules/lightgbm/plot_utils.py

import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class PlotUtils:
    def __init__(self):
        self.plot_dir = 'plots/lightgbm/'
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_learning_curve(self, evals_result, fold: int = None):
        plt.figure(figsize=(10, 6))
        plt.plot(evals_result['train']['multi_logloss'], label='Train RMSE')
        plt.plot(evals_result['validation']['multi_logloss'], label='Validation RMSE')
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Learning Curve')
        plt.legend()
        if fold is not None:
            plot_path = os.path.join(self.plot_dir, f'learning_curve_fold{fold}.png')
        else:
            plot_path = os.path.join(self.plot_dir, 'learning_curve.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"学習曲線を {plot_path} に保存しました。")

    def plot_feature_importance(self, model, feature_names, fold: int = None):
        importance = model.feature_importance()
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values(by='importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
        plt.title('Feature Importance')
        if fold is not None:
            plot_path = os.path.join(self.plot_dir, f'feature_importance_fold{fold}.png')
        else:
            plot_path = os.path.join(self.plot_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"特徴量重要度を {plot_path} に保存しました。")

    def plot_prediction(self, y_true, y_pred, fold: int = None):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        if fold is not None:
            plot_path = os.path.join(self.plot_dir, f'actual_vs_predicted_fold{fold}.png')
        else:
            plot_path = os.path.join(self.plot_dir, 'actual_vs_predicted.png')
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"予測結果を {plot_path} に保存しました。")

    def plot_confusion_matrix(self, y_true, y_pred, cm, fold: int = None):
        # 指標を計算
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        # プロットの設定
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # タイトルに指標を追加
        plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2f} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1 Score: {f1:.2f}')

        # ファイルパスの設定と保存
        if fold is not None:
            plot_path = os.path.join(self.plot_dir, f'confusion_matrix_fold{fold}.png')
        else:
            plot_path = os.path.join(self.plot_dir, 'confusion_matrix.png')
            
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"混同行列と指標を {plot_path} に保存しました。")