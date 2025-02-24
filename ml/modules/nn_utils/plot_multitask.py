import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np
from pathlib import Path

def plot_final_training_results_multitask(plot_dict: dict):
    # --- 保存先ディレクトリの作成 ---
    cv_plot_dir = Path("plots/rnn_base/multitask_model/cv")
    final_plot_dir = Path("plots/rnn_base/multitask_model/final")
    cv_plot_dir.mkdir(parents=True, exist_ok=True)
    final_plot_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================
    # 1. CV評価結果のプロット
    # ==========================
    # "result" キーがなければ "final_result" を利用する
    result = plot_dict.get("result", plot_dict.get("final_result", {}))
    eval_results = result.get("evaluation_results", {}).get("primary_model", {})
    
    if eval_results:
        multi_logloss = eval_results.get("multi_logloss", np.nan)
        test_rmse  = eval_results.get("test_rmse", np.nan)
        
        # CV評価結果にテストR²を追加（複数ターゲットの場合は平均値を表示）
        cv_evals = plot_dict.get("evals_result", {})
        test_r2 = np.nan
        if cv_evals:
            test_r2_dict = cv_evals.get("test_r2", {})
            if test_r2_dict:
                test_r2 = np.mean(list(test_r2_dict.values()))
        
        metrics = ["Multi Logloss", "Test R²", "Test RMSE"]
        values  = [multi_logloss, test_r2, test_rmse]
        
        fig_cv, ax_cv = plt.subplots(figsize=(6, 5))
        ax_cv.bar(metrics, values, color=["skyblue", "lightgreen", "salmon"])
        ax_cv.set_title("CV評価結果")
        for i, v in enumerate(values):
            ax_cv.text(i, v, f"{v:.3f}", ha='center', va='bottom')
        ax_cv.grid(True, axis="y")
        
        cv_plot_path = cv_plot_dir / "cv_results.png"
        fig_cv.tight_layout()
        fig_cv.savefig(cv_plot_path)
        print(f"CV結果のプロットを保存しました: {cv_plot_path}")
        plt.close(fig_cv)
    else:
        print("CV評価結果が存在しません。")
    
    # ============================================
    # 2. 最終学習時のエポック推移プロット（Loss, RMSE, R²）
    # ============================================
    # "final_result" 内に evals_result が含まれていない場合は、plot_dict 直下の evals_result を利用
    evals_result = result.get("evals_result", {}) or plot_dict.get("evals_result", {})
    
    if not evals_result:
        print("最終学習時の評価結果（evals_result）が存在しません。")
    else:
        train_loss = evals_result.get("train_loss", [])
        val_loss   = evals_result.get("val_loss", [])
        train_rmse = evals_result.get("train_rmse", [])
        val_rmse   = evals_result.get("val_rmse", [])
        # R²は学習時に記録していない場合もあるので、空の場合はその旨表示する
        train_r2   = evals_result.get("train_r2", [])
        val_r2     = evals_result.get("val_r2", [])
        epochs = range(1, len(train_loss) + 1)
        
        # 3サブプロット：Loss, RMSE, R²
        fig_final, axes_final = plt.subplots(1, 3, figsize=(21, 5))
        
        # Loss推移
        axes_final[0].plot(epochs, train_loss, marker="o", label="Train Loss")
        if val_loss:
            axes_final[0].plot(epochs, val_loss, marker="o", label="Validation Loss")
        axes_final[0].set_title("エポックごとのLoss推移")
        axes_final[0].set_xlabel("Epoch")
        axes_final[0].set_ylabel("Loss")
        axes_final[0].grid(True)
        axes_final[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
        # RMSE推移
        axes_final[1].plot(epochs, train_rmse, marker="o", label="Train RMSE")
        if val_rmse:
            axes_final[1].plot(epochs, val_rmse, marker="o", label="Validation RMSE")
        axes_final[1].set_title("エポックごとのRMSE推移")
        axes_final[1].set_xlabel("Epoch")
        axes_final[1].set_ylabel("RMSE")
        axes_final[1].grid(True)
        axes_final[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
        # R²推移（学習時に記録されていない場合はメッセージを表示）
        if train_r2 and val_r2:
            axes_final[2].plot(epochs, train_r2, marker="o", label="Train R²")
            axes_final[2].plot(epochs, val_r2, marker="o", label="Validation R²")
        else:
            axes_final[2].text(0.5, 0.5, "R²の評価結果がありません", ha="center", va="center", fontsize=12)
        axes_final[2].set_title("エポックごとのR²推移")
        axes_final[2].set_xlabel("Epoch")
        axes_final[2].set_ylabel("R² Score")
        axes_final[2].grid(True)
        if train_r2 and val_r2:
            axes_final[2].legend(loc="center left", bbox_to_anchor=(1, 0.5))
        
        fig_final.suptitle("全データを用いた最終学習の経過", fontsize=16)
        fig_final.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        final_plot_path = final_plot_dir / "final_training_results.png"
        fig_final.savefig(final_plot_path)
        print(f"最終学習結果のプロットを保存しました: {final_plot_path}")
        plt.close(fig_final)
    
    # =====================================================
    # 3. テストデータの予測結果評価（散布図＋残差プロット）
    # =====================================================
    # マルチタスクの場合、ターゲットごとに保存しているため、各ターゲットごとにプロット
    evals_result = result.get("evals_result", {}) or plot_dict.get("evals_result", {})
    test_labels_by_target = evals_result.get("test_labels_by_target", {})
    test_preds_by_target  = evals_result.get("test_preds_by_target", {})
    
    if test_labels_by_target and test_preds_by_target:
        for target, labels_list in test_labels_by_target.items():
            preds_list = test_preds_by_target.get(target, [])
            if labels_list and preds_list:
                labels = np.array(labels_list)
                preds = np.array(preds_list)
                
                fig_test, axes_test = plt.subplots(1, 2, figsize=(12, 6))
                
                # 散布図：予測値 vs. 正解値
                axes_test[0].scatter(labels, preds, alpha=0.7, label='データ点')
                min_val = min(labels.min(), preds.min())
                max_val = max(labels.max(), preds.max())
                axes_test[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想線 (y=x)')
                axes_test[0].set_xlabel("正解値")
                axes_test[0].set_ylabel("予測値")
                axes_test[0].set_title(f"{target}の予測値 vs. 正解値")
                axes_test[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
                axes_test[0].grid(True)
                
                # 残差プロット：残差 = 正解値 - 予測値
                residuals = labels - preds
                axes_test[1].scatter(preds, residuals, alpha=0.7)
                axes_test[1].axhline
