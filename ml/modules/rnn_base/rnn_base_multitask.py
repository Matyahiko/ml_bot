# -*- coding: utf-8 -*-
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import pandas as pd
import logging
from rich import print
import numpy as np
import pickle  # 結果保存用に追加

# Optuna 関連
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import Trial

# ユーザー定義の前処理済みデータセットなど
from modules.rnn_base.rnn_data_process_multitask import TimeSeriesDataset

#plot
from modules.nn_utils.plot_multitask import  plot_final_training_results_multitask
from sklearn.metrics import r2_score

# ロギング設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

# =================================
# シンプルなGRUモデル例（多変量回帰用）
# =================================
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=1, dropout_rate=0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # シーケンス最後の時刻の出力を使用
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

# =================================
# RNNBase クラス（学習・検証など）
# =================================
class RNNBase(nn.Module):
    def __init__(self, **kwargs):
        """
        Parameters:
            train_df : pd.DataFrame
            test_df  : pd.DataFrame
            target_columns : list of str （例：["target_vol", "target_price"]）
            device   : str (例："cuda:0")
        """
        super().__init__()
        self.train_df = kwargs.get('train_df')
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df')
        
        # target_columnsをリストに統一
        target = kwargs.get('target_columns')
        self.target_columns = target if isinstance(target, list) else [target]

        self.device = kwargs.get('device', 'cuda:0')

        # 初期ハイパーパラメータ
        self.batch_size = 128
        self.learning_rate = 0.0006
        self.num_epochs = 25
        self.hidden_size = 512
        self.num_layers = 1
        self.window_size = 50
        self.dropout_rate = 0.1
        # strideのデフォルト値を追加（例：25）
        self.stride = 10

        self.gradient_accumulation_steps = 30
        self.early_stopping_patience = 3
        self.do_hyperparam_search = True
        self.is_hyperparam_search = False  # ハイパラ探索中かどうかのフラグ

        # 評価用の途中経過は最終学習時のみ保存するためのフラグ（初期はFalse）
        self.save_eval_progress = False

        self.data_set = TimeSeriesDataset(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            target_columns=self.target_columns,  # マルチタスク用
            window_size=self.window_size,
            stride=self.stride  # 追加
        )

        # 必要な評価値のみ保存する（学習曲線・残差プロット・R²スコアのため）
        self.evals_result = {
            'train_loss': [],
            'train_rmse': [],
            'val_loss': [],
            'val_rmse': [],
            'test_labels_by_target': None,
            'test_preds_by_target': None,
            'test_r2': None
        }

        self.best_val_loss = float('inf')
        self.best_hyperparams = {}

    def run(self, fold: int = None, gpu_queue=None):
        # 最終学習時のみ途中結果を保存する
        if fold is None and not self.is_hyperparam_search:
            self.save_eval_progress = True
        else:
            self.save_eval_progress = False

        if self.do_hyperparam_search:
            self.is_hyperparam_search = True

            def objective(trial: Trial):
                # GPU利用時はgpu_queueからIDを取得（なければself.deviceを利用）
                device = self.device
                if gpu_queue is not None:
                    gpu_id = gpu_queue.get()
                    device = f"cuda:{gpu_id}"
                    print(f"Trial {trial.number} using device {device}")

                params = {
                    'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
                    'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-2),
                    'hidden_size': trial.suggest_int('hidden_size', 512, 1024, step=32),
                    'num_layers': trial.suggest_int('num_layers', 1, 2),
                    'window_size': trial.suggest_int('window_size', 10, 100, step=10),
                    'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.7, step=0.1),
                    'stride': trial.suggest_int('stride', 20, 100, step=10)  # 追加
                }
                num_epochs = 12

                # trial用のモデルインスタンスを生成
                trial_model = RNNBase(
                    train_df=self.train_df,
                    val_df=self.val_df,
                    test_df=self.test_df,
                    target_columns=self.target_columns,
                    device=device
                )
                trial_model.is_hyperparam_search = True

                trial_model.batch_size = params['batch_size']
                trial_model.learning_rate = params['learning_rate']
                trial_model.hidden_size = params['hidden_size']
                trial_model.num_layers = params['num_layers']
                trial_model.num_epochs = num_epochs
                trial_model.window_size = params['window_size']
                trial_model.dropout_rate = params['dropout_rate']
                trial_model.stride = params['stride']  # 追加

                # データセットを再構築（strideも更新）
                trial_model.data_set = TimeSeriesDataset(
                    train_df=self.train_df,
                    val_df=self.val_df,
                    test_df=self.test_df,
                    target_columns=self.target_columns,
                    window_size=params['window_size'],
                    stride=params['stride']  # 追加
                )

                result = trial_model._train_and_evaluate()
                if gpu_queue is not None:
                    gpu_queue.put(gpu_id)
                return result.get("val_loss", float('inf'))

            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=2)
            )
            # 並列処理をやめ、シーケンシャルに探索（n_jobs指定を削除）
            study.optimize(objective, n_trials=30)

            best_params = study.best_trial.params
            self.batch_size = best_params['batch_size']
            self.learning_rate = best_params['learning_rate']
            self.hidden_size = best_params['hidden_size']
            self.num_layers = best_params['num_layers']
            self.window_size = best_params['window_size']
            self.dropout_rate = best_params['dropout_rate']
            self.stride = best_params['stride']  # 追加
            self.num_epochs = 10

            print(f"ベストハイパーパラメータ: {best_params}")

            self.data_set = TimeSeriesDataset(
                train_df=self.train_df,
                val_df=self.val_df,
                test_df=self.test_df,
                target_columns=self.target_columns,
                window_size=self.window_size,
                stride=self.stride  # 追加
            )
            self.is_hyperparam_search = False

        result_metrics = self._train_and_evaluate(fold=fold)
        return {
            'evaluation_results': {
                'primary_model': {
                    'multi_logloss': result_metrics.get("test_loss", None),
                    'test_rmse': result_metrics.get("test_rmse", None)
                }
            }
        }

    def _train_and_evaluate(self, fold=None):
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        train_dataset = TensorDataset(self.data_set.train_features, self.data_set.train_labels)
        val_dataset = TensorDataset(self.data_set.val_features, self.data_set.val_labels) if self.data_set.val_features is not None else None
        test_dataset = TensorDataset(self.data_set.test_features, self.data_set.test_labels) if self.data_set.test_features is not None else None

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1) if val_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1) if test_dataset else None

        sample_x, _ = next(iter(train_loader))
        feature_dim = sample_x.shape[-1]

        model = GRUModel(
            input_size=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=len(self.target_columns),
            dropout_rate=self.dropout_rate
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = 0
        no_improve_count = 0

        model_save_dir = Path("models/rnn_base")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        target_str = "_".join(self.target_columns)

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            total = 0
            train_preds_list = []
            train_labels_list = []
            optimizer.zero_grad()

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                raw_loss = criterion(outputs, labels)
                (raw_loss / self.gradient_accumulation_steps).backward()

                running_loss += raw_loss.item() * inputs.size(0)
                total += labels.size(0)
                # 予測値と正解値を保存
                train_preds_list.append(outputs.detach().cpu())
                train_labels_list.append(labels.detach().cpu())
                # 途中経過のlossも記録（必要であれば）
                if self.save_eval_progress:
                    self.evals_result.setdefault('iteration_loss', []).append(raw_loss.item())

                if (i + 1) % self.gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss = running_loss / total
            epoch_rmse = np.sqrt(epoch_loss)
            if self.save_eval_progress:
                self.evals_result['train_loss'].append(epoch_loss)
                self.evals_result['train_rmse'].append(epoch_rmse)
                # 学習時の R² を計算
                all_train_preds = torch.cat(train_preds_list)
                all_train_labels = torch.cat(train_labels_list)
                epoch_train_r2 = r2_score(all_train_labels.numpy(), all_train_preds.numpy())
                self.evals_result.setdefault("train_r2", []).append(epoch_train_r2)

            # 検証処理
            if val_loader:
                model.eval()
                val_running_loss = 0.0
                val_total = 0
                val_preds_list = []
                val_labels_list = []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device).float()
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_running_loss += loss.item() * inputs.size(0)
                        val_total += labels.size(0)
                        val_preds_list.append(outputs.detach().cpu())
                        val_labels_list.append(labels.detach().cpu())

                val_epoch_loss = val_running_loss / val_total
                val_epoch_rmse = np.sqrt(val_epoch_loss)
                if self.save_eval_progress:
                    self.evals_result['val_loss'].append(val_epoch_loss)
                    self.evals_result['val_rmse'].append(val_epoch_rmse)
                    # 検証時の R² を計算
                    all_val_preds = torch.cat(val_preds_list)
                    all_val_labels = torch.cat(val_labels_list)
                    epoch_val_r2 = r2_score(all_val_labels.numpy(), all_val_preds.numpy())
                    self.evals_result.setdefault("val_r2", []).append(epoch_val_r2)
            else:
                val_epoch_loss = None

            # 最終学習時のみ各エポックごとにチェックポイントを保存
            if fold is None and not self.is_hyperparam_search:
                checkpoint_path = model_save_dir / f"rnn_model_{target_str}_final_epoch{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_loss,
                    'val_loss': val_epoch_loss,
                    'train_rmse': epoch_rmse,
                    'val_rmse': val_epoch_rmse,
                    'hyperparameters': {
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate,
                        'hidden_size': self.hidden_size,
                        'num_layers': self.num_layers,
                        'window_size': self.window_size,
                        'dropout_rate': self.dropout_rate,
                        'num_epochs': self.num_epochs,
                        'gradient_accumulation_steps': self.gradient_accumulation_steps,
                        'early_stopping_patience': self.early_stopping_patience,
                        'stride': self.stride
                    }
                }, checkpoint_path)
                print(f"Epoch {epoch+1} checkpoint saved: {checkpoint_path}")

            if val_loader:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs. Best epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")
                    break

        model.load_state_dict(best_model_wts)

        # テスト評価部分（既存のコード）
        test_loss = test_rmse = None
        if test_loader:
            model.eval()
            test_running_loss = 0.0
            test_total = 0
            test_preds_list = []
            test_labels_list = []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_running_loss += loss.item() * inputs.size(0)
                    test_total += labels.size(0)
                    test_preds_list.append(outputs.detach().cpu())
                    test_labels_list.append(labels.detach().cpu())

            test_loss = test_running_loss / test_total
            test_rmse = np.sqrt(test_loss)

            all_test_preds = torch.cat(test_preds_list)
            all_test_labels = torch.cat(test_labels_list)
            all_test_preds_np = all_test_preds.numpy()
            all_test_labels_np = all_test_labels.numpy()

            test_r2_scores = {}
            test_labels_by_target = {}
            test_preds_by_target = {}

            for i, target in enumerate(self.target_columns):
                true_i = all_test_labels_np[:, i]
                pred_i = all_test_preds_np[:, i]
                r2_i = r2_score(true_i, pred_i)
                test_r2_scores[target] = r2_i
                test_labels_by_target[target] = true_i.tolist()
                test_preds_by_target[target] = pred_i.tolist()

            if self.save_eval_progress:
                self.evals_result['test_labels_by_target'] = test_labels_by_target
                self.evals_result['test_preds_by_target'] = test_preds_by_target
                self.evals_result['test_r2'] = test_r2_scores

            # print("=== テスト評価結果（ターゲット別） ===")
            # for target in self.target_columns:
            #     print(f"{target}: Loss = {test_loss:.4f}, R² = {test_r2_scores[target]:.4f}, RMSE = {test_rmse:.4f}")

        self.best_val_loss = best_val_loss
        self.best_hyperparams = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'window_size': self.window_size,
            'dropout_rate': self.dropout_rate,
            'num_epochs': self.num_epochs,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'stride': self.stride
        }

        return {"val_loss": best_val_loss, "test_loss": test_loss, "test_rmse": test_rmse}


# ===============================
# メイン処理例
# ===============================
if __name__ == '__main__':
    # 例：train_df, val_df, test_df, target_columnsは各自実データで定義
    train_df = pd.DataFrame()  # ここに実データ読み込み処理
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    target_columns = ["target_price"]

    # RNNBaseのインスタンス生成（デフォルトはシーケンシャル探索）
    model_instance = RNNBase(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_columns=target_columns,
        device='cuda:0'
    )

    # gpu_queueなどの並列処理関連は削除し、シーケンシャルに実行
    results = model_instance.run()
    print("最終評価結果:", results)


# =================================
# 以下、foldごとの学習・評価処理など（基本的に変更なし）
# =================================
def load_data(file_path: Path) -> pd.DataFrame:
    logger.info(f"データ読み込み: {file_path}")
    data = pd.read_pickle(file_path)
    logger.info(f"データ読み込み完了: {file_path}")
    object_columns = data.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        logger.warning(f"object型の列が存在: {object_columns}")
    return data

def get_file_paths(data_dir: Path, filename: str, fold: int) -> (Path, Path):
    train_file = data_dir / f"{filename}_fold{fold}_train.pkl"
    test_file = data_dir / f"{filename}_fold{fold}_test.pkl"
    return train_file, test_file

def train_model_for_fold(fold: int, config: dict) -> dict:
    logger.info(f"フォールド {fold} の学習開始")
    data_dir = config["data_dir"]
    filename = config["filename"]
    train_file, test_file = get_file_paths(data_dir, filename, fold)
    
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    
    train_df = train_df.drop(columns=config["exclude_target"])
    test_df = test_df.drop(columns=config["exclude_target"])
    
    device = config["device"]
    model = RNNBase(
        train_df=train_df,
        test_df=test_df,
        target_columns=config["target_columns"],
        device=device
    )
    result = model.run(fold)
    logger.info(f"フォールド {fold} の学習完了。結果: {result}")
    return {
        "fold": fold,
        "result": result,
        "evals_result": model.evals_result,
        "best_hyperparams": model.best_hyperparams
    }

def aggregate_results(results: list):
    if not results:
        logger.warning("集約する結果がありません。")
        return
    total_loss = total_rmse = 0
    print("\n==== フォールド別結果 ====")
    for res in results:
        fold = res["fold"]
        performance = res["result"]["evaluation_results"]["primary_model"]
        test_loss = performance.get("multi_logloss", None)
        test_rmse = performance.get("test_rmse", None)
        
        print(f"フォールド {fold}: テスト損失 = {test_loss:.4f}, RMSE = {test_rmse:.4f}")
        total_loss += test_loss
        total_rmse += test_rmse
    count = len(results)
    print("-----------------------------")
    print(f"全体平均テスト損失: {total_loss/count:.4f}")
    print(f"全体平均RMSE: {total_rmse/count:.4f}\n")

def train_model_on_full_data(config: dict, final_hyperparams: dict) -> dict:
    logger.info("全データを用いた最終学習を開始")
    data_dir = config["data_dir"]
    filename = config["filename"]

    full_file = data_dir / f"{filename}_train.pkl"
    test_file = data_dir / f"{filename}_test.pkl"

    full_df = load_data(full_file)
    test_df = load_data(test_file)

    full_df = full_df.drop(columns=config["exclude_target"])
    test_df = test_df.drop(columns=config["exclude_target"])

    device = config["device"]

    model = RNNBase(
        train_df=full_df,
        test_df=test_df,
        target_columns=config["target_columns"],
        device=device
    )

    model.batch_size = final_hyperparams["batch_size"]
    model.learning_rate = final_hyperparams["learning_rate"]
    model.hidden_size = final_hyperparams["hidden_size"]
    model.num_layers = final_hyperparams["num_layers"]
    model.window_size = final_hyperparams["window_size"]
    model.dropout_rate = final_hyperparams["dropout_rate"]
    model.num_epochs = final_hyperparams["num_epochs"]
    model.gradient_accumulation_steps = final_hyperparams["gradient_accumulation_steps"]
    model.early_stopping_patience = final_hyperparams["early_stopping_patience"]
    model.stride = final_hyperparams["stride"]  # 追加

    # 再度データセットを構築（window_size, strideを反映）
    model.data_set = TimeSeriesDataset(
        train_df=full_df,
        val_df=None,
        test_df=test_df,
        target_columns=config["target_columns"],
        window_size=model.window_size,
        stride=model.stride  # 追加
    )

    final_result_metrics = model.run()  
    final_result_metrics["evals_result"] = model.evals_result

    results_save_dir = Path("storage/predictions")
    results_save_dir.mkdir(parents=True, exist_ok=True)
    target_str = "_".join(config["target_columns"])
    results_save_path = results_save_dir / f"rnn_test_results_{target_str}.pkl"
    with open(results_save_path, "wb") as f:
        pickle.dump(final_result_metrics, f)
    print(f"最終テスト結果を {results_save_path} に保存しました")

    logger.info("全データを用いた最終学習完了")
    return {
        "result": final_result_metrics,
        "evals_result": model.evals_result,
        "best_hyperparams": model.best_hyperparams
    }

def run_folds(config: dict):
    results = []
    num_folds = config.get("num_folds", 5)
    for fold in range(num_folds):
        res = train_model_for_fold(fold, config)
        results.append(res)
    aggregate_results(results)
    logger.info("全フォールドの学習完了。")

    best_fold = min(results, key=lambda x: x["result"]["evaluation_results"]["primary_model"]["multi_logloss"])
    final_hyperparams = best_fold["best_hyperparams"]
    print("選択された最良ハイパーパラメータ:", final_hyperparams)

    final_training_result = train_model_on_full_data(config, final_hyperparams)
    print("最終的な全データによる再学習の結果:")
    print(final_training_result)

    plot = {
        "fold_results": results,
        "final_result": final_training_result
    }
    
    plot_final_training_results_multitask(plot)
