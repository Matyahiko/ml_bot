import os
import matplotlib
# 非表示モードにする場合、'Agg'バックエンドを使用する方法もあります
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_epoch_learning_curve(evals_result, save_path=None):
    epochs = range(1, len(evals_result.get('train_loss', [])) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, evals_result.get('train_loss', []), label='Train Loss', marker='o')
    if evals_result.get('val_loss'):
        plt.plot(epochs, evals_result.get('val_loss'), label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epoch-wise Learning Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    # 画像を表示せずに閉じる
    plt.close()

def plot_epoch_accuracy_curve(evals_result, save_path=None):
    epochs = range(1, len(evals_result.get('train_acc', [])) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, evals_result.get('train_acc', []), label='Train R2', marker='o')
    if evals_result.get('val_acc'):
        plt.plot(epochs, evals_result.get('val_acc'), label='Validation R2', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.title('Epoch-wise R2 Score Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_iteration_learning_curve(evals_result, save_path=None):
    if not evals_result.get('iteration_loss'):
        print("Iteration loss data is not available.")
        return
    iterations = range(1, len(evals_result['iteration_loss']) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, evals_result['iteration_loss'], label='Iteration Loss', marker='.', linestyle='-')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Iteration-wise Learning Curve')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_model_summary_table(best_hyperparams, save_path=None):
    df = pd.DataFrame(list(best_hyperparams.items()), columns=['Parameter', 'Value'])
    fig, ax = plt.subplots(figsize=(6, len(df)*0.4+1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Optimal Model Hyperparameters', fontweight="bold")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_fold_r2_table(fold_results, save_path=None):
    # 各フォールドの結果から、fold番号と R2（accuracy） を抽出
    data = []
    for res in fold_results:
        fold = res.get("fold")
        r2 = res.get("result", {}).get("evaluation_results", {})\
                    .get("primary_model", {}).get("accuracy", None)
        # -∞ もしくは極端に低い値の場合は –1 にする（表示上の工夫）
        if r2 is None or (isinstance(r2, float) and (np.isneginf(r2) or r2 < -1)):
            r2 = -1
        data.append({"Fold": fold, "R2 Score": r2})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots(figsize=(4, len(df)*0.5+1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    ax.set_title('Fold-wise R2 Scores', fontweight="bold")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_prediction_vs_actual(evals_result, save_path=None):
    test_labels = evals_result.get("test_labels")
    test_preds = evals_result.get("test_preds")
    if test_labels is None or test_preds is None:
        print("Test labels or predictions are not available.")
        return
    plt.figure(figsize=(10, 5))
    plt.plot(test_labels, label="Actual", marker="o", linestyle="-")
    plt.plot(test_preds, label="Predicted", marker="o", linestyle="-")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("Test Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_prediction_vs_actual_distribution(evals_result, save_path=None):
    """
    テストデータの正解値と予測値の分布をヒストグラムとして重ねて表示します。
    """
    test_labels = evals_result.get("test_labels")
    test_preds = evals_result.get("test_preds")
    if test_labels is None or test_preds is None:
        print("Test labels or predictions are not available.")
        return
    plt.figure(figsize=(10, 5))
    bins = 30  # ヒストグラムのビン数は必要に応じて変更してください
    plt.hist(test_labels, bins=bins, alpha=0.5, density=True, label="Actual")
    plt.hist(test_preds, bins=bins, alpha=0.5, density=True, label="Predicted")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Distribution of Test Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_all_results(evals_result, best_hyperparams, fold_results, save_dir=None):
    """
    各プロットを順次実行するマスタ関数
      - エポックごとの損失・R2曲線
      - イテレーションごとの損失曲線
      - 最適ハイパーパラメータの表
      - フォールドごとの R2 表
      - 予測値と正解値の系列プロット
      - 予測値と正解値の分布プロット
    save_dir が指定されていれば、各図を固定のファイル名で保存します。
    """
    if save_dir:
        save_dir = str(save_dir)
        loss_curve_path = os.path.join(save_dir, "epoch_loss_curve.png")
        r2_curve_path = os.path.join(save_dir, "epoch_r2_curve.png")
        iteration_curve_path = os.path.join(save_dir, "iteration_loss_curve.png")
        model_summary_path = os.path.join(save_dir, "model_summary_table.png")
        fold_r2_table_path = os.path.join(save_dir, "fold_r2_table.png")
        prediction_vs_actual_path = os.path.join(save_dir, "prediction_vs_actual.png")
        prediction_distribution_path = os.path.join(save_dir, "prediction_distribution.png")
    else:
        loss_curve_path = r2_curve_path = iteration_curve_path = model_summary_path = fold_r2_table_path = prediction_vs_actual_path = prediction_distribution_path = None

    plot_epoch_learning_curve(evals_result, save_path=loss_curve_path)
    plot_epoch_accuracy_curve(evals_result, save_path=r2_curve_path)
    plot_iteration_learning_curve(evals_result, save_path=iteration_curve_path)
    plot_model_summary_table(best_hyperparams, save_path=model_summary_path)
    plot_fold_r2_table(fold_results, save_path=fold_r2_table_path)
    plot_prediction_vs_actual(evals_result, save_path=prediction_vs_actual_path)
    plot_prediction_vs_actual_distribution(evals_result, save_path=prediction_distribution_path)
