import numpy as np
import pandas as pd
import pathlib as path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=None):
    """
    混同行列を表示する
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='.2f' if normalize else 'd')
    plt.title('Confusion Matrix')
    plt.savefig('plots/cnn_base/confusion_matrix.png')
    


def plot_learning_curves(train_loss, val_loss, train_acc, val_acc):
    """
    学習曲線(損失/精度)を表示する
    """
    epochs = range(1, len(train_loss) + 1)

    # --- Loss ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Train Loss')
    if len(val_loss) > 0:
        plt.plot(epochs, val_loss, 'ro-', label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # --- Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Train Acc')
    if len(val_acc) > 0:
        plt.plot(epochs, val_acc, 'ro-', label='Val Acc')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/cnn_base/learning_curves.png')


def plot_roc_auc(y_true, y_probas, class_names=None):
    """
    ROC曲線とAUCを描画する (マルチクラス対応)
    y_probas: shape = (num_samples, n_classes)
    """
    n_classes = len(np.unique(y_true))  # 実際のクラス数
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    # 各クラスのROCを計算
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_probas[:, i])
        roc_auc = auc(fpr, tpr)
        label = (f"Class {class_names[i]} (AUC={roc_auc:.2f})"
                 if class_names else f"Class {i} (AUC={roc_auc:.2f})")
        plt.plot(fpr, tpr, label=label)

    # マイクロ平均 (全クラスをまとめた指標) を計算
    fpr_micro, tpr_micro, _ = roc_curve(y_true_binarized.ravel(), y_probas.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro,
             label=f"Micro-average (AUC={roc_auc_micro:.2f})",
             linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig('plots/cnn_base/roc_auc.png')


def plot_roc_auc_binary(y_true, y_probas):
    """
    二値分類用のROC曲線とAUCを描画する
    y_probas: shape = (num_samples,) または (num_samples, 1)
    """
    if y_probas.ndim == 2 and y_probas.shape[1] == 1:
        y_probas = y_probas.ravel()

    fpr, tpr, _ = roc_curve(y_true, y_probas)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary)")
    plt.legend(loc="lower right")
    plt.savefig('plots/cnn_base/roc_auc_binary.png')


def plot_all_results(evals_result, class_names=None):
    """
    evals_result に格納された情報をまとめて可視化する
    - 学習曲線 (train_loss, val_loss, train_acc, val_acc)
    - 混同行列 (test_labels, test_preds)
    - ROC-AUC (test_labels, test_probas)
    """
    train_loss = evals_result.get('train_loss', [])
    val_loss = evals_result.get('val_loss', [])
    train_acc = evals_result.get('train_acc', [])
    val_acc = evals_result.get('val_acc', [])
    test_labels = evals_result.get('test_labels', [])
    test_preds = evals_result.get('test_preds', [])
    test_probas = evals_result.get('test_probas', [])

    # --------- ここから追加の修正: one-hot → 1次元へ ---------
    test_labels = np.array(test_labels)
    test_preds = np.array(test_preds)
    test_probas = np.array(test_probas)

    # (A) ラベルを 1D に変換
    if test_labels.ndim == 2:
        test_labels = np.argmax(test_labels, axis=1)

    # (B) 離散予測も 1D に変換
    #     もし test_preds に確率が入っているなら argmax する。  
    #     あるいはバイナリクラス分類なら (test_preds>0.5) など。
    if test_preds.ndim == 2:
        test_preds = np.argmax(test_preds, axis=1)
    # ------------------------------------------------------

    # 1. 学習曲線をプロット
    plot_learning_curves(train_loss, val_loss, train_acc, val_acc)

    # 2. 混同行列 (ラベル/予測は1Dであることが必要)
    if len(test_labels) > 0 and len(test_preds) > 0:
        plot_confusion_matrix(test_labels, test_preds,
                              class_names=class_names, normalize=None)

    # 3. ROC-AUC
    #   二値分類か多クラスかを判断して分岐
    unique_labels = np.unique(test_labels)
    if len(unique_labels) == 2:
        # 二値分類
        if len(test_labels) > 0 and len(test_probas) > 0:
            # test_probas が (N,2) なら、陽性クラスの列を取り出すか、
            # あるいは二値用のロジックに合わせて変換:
            if test_probas.ndim == 2 and test_probas.shape[1] == 2:
                # 例えば陽性クラス(クラスID=1)の確率だけ抽出
                test_probas_binary = test_probas[:, 1]
                plot_roc_auc_binary(test_labels, test_probas_binary)
            else:
                # (N,) or (N,1)の場合など
                plot_roc_auc_binary(test_labels, test_probas)
    else:
        # マルチクラスの場合
        if len(test_labels) > 0 and len(test_probas) > 0:
            # (N, クラス数) が想定
            plot_roc_auc(test_labels, test_probas, class_names=class_names)
