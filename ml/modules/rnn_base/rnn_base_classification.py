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
from modules.rnn_base.rnn_data_process import TimeSeriesDataset

# ロギング設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

# =================================
# シンプルなGRU分類モデル例（多クラス分類用）
# =================================
class SimpleGRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=2, dropout_rate=0.0):
        """
        Parameters:
            input_size  : 入力特徴量の次元数
            hidden_size : GRUの隠れ層ユニット数
            num_layers  : GRU層の層数
            num_classes : 分類クラス数
            dropout_rate: ドロップアウト率
        """
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
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        out = out[:, -1, :]  # 最後のタイムステップの出力を使用
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

# =================================
# RNNBase クラス（学習・検証など：分類版）
# =================================
class RNNBase(nn.Module):
    def __init__(self, **kwargs):
        """
        Parameters:
            train_df     : pd.DataFrame
            test_df      : pd.DataFrame
            target_column: str（ラベル列。クラスインデックスになっている前提）
            device       : str (例："cuda:0")
            num_classes  : int（分類クラス数、デフォルトは2）
        """
        super().__init__()
        self.train_df = kwargs.get('train_df')
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df')
        self.target_column = kwargs.get('target_column')
        self.device = kwargs.get('device', 'cuda:0')
        self.num_classes = kwargs.get('num_classes', 2)

        # ハイパーパラメータ（必要に応じて変更）
        self.batch_size = 64
        self.learning_rate = 0.0006089016654730252
        self.num_epochs = 10
        self.hidden_size = 256
        self.num_layers = 4
        self.window_size = 90
        self.dropout_rate = 0.0

        self.gradient_accumulation_steps = 200
        self.early_stopping_patience = 3
        self.do_hyperparam_search = True

        self.data_set = TimeSeriesDataset(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            target_column=self.target_column,
            window_size=self.window_size
        )

        # 記録用（評価指標は損失と正解率）
        self.evals_result = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'iteration_loss': []  
        }

        self.best_val_loss = float('inf')
        self.best_hyperparams = {}

    def run(self, fold: int = None):
        if self.do_hyperparam_search:
            def objective(trial: Trial):
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
                hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
                num_layers = trial.suggest_int('num_layers', 1, 4, step=1)
                window_size = trial.suggest_int('window_size', 10, 100, step=10)
                dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5, step=0.1)
                num_epochs = 5  # ハイパラ探索時は短めに

                # 元のパラメータを保持
                orig_params = (self.batch_size, self.learning_rate, self.hidden_size,
                               self.num_layers, self.num_epochs, self.window_size, self.dropout_rate)
                self.batch_size = batch_size
                self.learning_rate = learning_rate
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.num_epochs = num_epochs
                self.window_size = window_size
                self.dropout_rate = dropout_rate

                self.data_set = TimeSeriesDataset(
                    train_df=self.train_df,
                    val_df=self.val_df,
                    test_df=self.test_df,
                    target_column=self.target_column,
                    window_size=self.window_size
                )

                result = self._train_and_evaluate()
                val_loss = result.get("val_loss", float('inf'))

                (self.batch_size, self.learning_rate, self.hidden_size,
                 self.num_layers, self.num_epochs, self.window_size, self.dropout_rate) = orig_params

                return val_loss

            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=2)
            )
            study.optimize(objective, n_trials=100)

            best_params = study.best_trial.params
            self.batch_size = best_params['batch_size']
            self.learning_rate = best_params['learning_rate']
            self.hidden_size = best_params['hidden_size']
            self.num_layers = best_params['num_layers']
            self.window_size = best_params['window_size']
            self.dropout_rate = best_params['dropout_rate']
            self.num_epochs = 5

            print(f"ベストハイパーパラメータ: {best_params}")

            self.data_set = TimeSeriesDataset(
                train_df=self.train_df,
                val_df=self.val_df,
                test_df=self.test_df,
                target_column=self.target_column,
                window_size=self.window_size
            )

        result_metrics = self._train_and_evaluate(fold=fold)
        return {
            'evaluation_results': {
                'primary_model': {
                    'test_loss': result_metrics.get("test_loss", None),
                    'accuracy': result_metrics.get("test_acc", None)
                }
            }
        }

    def _train_and_evaluate(self, fold=None):
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        train_dataset = TensorDataset(self.data_set.train_features, self.data_set.train_labels)
        val_dataset = None
        if self.data_set.val_features is not None:
            val_dataset = TensorDataset(self.data_set.val_features, self.data_set.val_labels)
        test_dataset = None
        if self.data_set.test_features is not None:
            test_dataset = TensorDataset(self.data_set.test_features, self.data_set.test_labels)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1) if val_dataset is not None else None
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1) if test_dataset is not None else None

        sample_x, _ = next(iter(train_loader))
        feature_dim = sample_x.shape[-1]

        model = SimpleGRUClassifier(
            input_size=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = 0
        no_improve_count = 0

        # モデル保存用ディレクトリ（最終学習時のみチェックポイント保存）
        model_save_dir = Path("models/rnn_base")
        model_save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0.0
            total_samples = 0

            train_preds_list = []
            train_labels_list = []

            accumulation_steps = self.gradient_accumulation_steps
            optimizer.zero_grad()

            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                # 分類問題なので、ラベルはlong型に変換
                labels = labels.to(device).long()

                outputs = model(inputs)  # logits: (batch, num_classes)
                loss = criterion(outputs, labels)
                loss_div = loss / accumulation_steps
                loss_div.backward()

                running_loss += loss.item() * inputs.size(0)
                total_samples += labels.size(0)

                # 予測クラスを取得
                preds = torch.argmax(outputs, dim=1)
                train_preds_list.append(preds.detach().cpu())
                train_labels_list.append(labels.detach().cpu())

                self.evals_result['iteration_loss'].append(loss.item())

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss = running_loss / total_samples
            self.evals_result['train_loss'].append(epoch_loss)

            all_train_preds = torch.cat(train_preds_list)
            all_train_labels = torch.cat(train_labels_list)
            train_acc = (all_train_preds == all_train_labels).float().mean().item()
            self.evals_result['train_acc'].append(train_acc)

            # 検証処理（val_loaderがある場合）
            if val_loader is not None:
                model.eval()
                val_running_loss = 0.0
                val_total = 0
                val_preds_list = []
                val_labels_list = []

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device).long()

                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                        val_running_loss += loss.item() * inputs.size(0)
                        val_total += labels.size(0)

                        preds = torch.argmax(outputs, dim=1)
                        val_preds_list.append(preds.detach().cpu())
                        val_labels_list.append(labels.detach().cpu())

                val_epoch_loss = val_running_loss / val_total
                self.evals_result['val_loss'].append(val_epoch_loss)

                all_val_preds = torch.cat(val_preds_list)
                all_val_labels = torch.cat(val_labels_list)
                val_acc = (all_val_preds == all_val_labels).float().mean().item()
                self.evals_result['val_acc'].append(val_acc)
            else:
                val_epoch_loss = None
                val_acc = None

            # -----------------------------
            # 【最終学習時のみ】各エポック終了時にチェックポイントとして保存
            # -----------------------------
            if fold is None:
                checkpoint_path = model_save_dir / f"rnn_classifier_{self.target_column}_final_epoch{epoch+1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_loss,
                    'val_loss': val_epoch_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'hyperparameters': {
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate,
                        'hidden_size': self.hidden_size,
                        'num_layers': self.num_layers,
                        'window_size': self.window_size,
                        'dropout_rate': self.dropout_rate,
                        'num_epochs': self.num_epochs,
                        'gradient_accumulation_steps': self.gradient_accumulation_steps,
                        'early_stopping_patience': self.early_stopping_patience
                    }
                }, checkpoint_path)
                print(f"Epoch {epoch+1} checkpoint saved: {checkpoint_path}")

            # 早期終了判定（検証データがある場合）
            if val_loader is not None:
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                if no_improve_count >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs. Best epoch was {best_epoch} with Val Loss: {best_val_loss:.4f}")
                    break

        model.load_state_dict(best_model_wts)

        test_loss = None
        test_acc = None
        if test_loader is not None:
            model.eval()
            test_running_loss = 0.0
            test_total = 0

            test_preds_list = []
            test_labels_list = []

            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device).long()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    test_running_loss += loss.item() * inputs.size(0)
                    test_total += labels.size(0)

                    preds = torch.argmax(outputs, dim=1)
                    test_preds_list.append(preds.detach().cpu())
                    test_labels_list.append(labels.detach().cpu())

            test_loss = test_running_loss / test_total
            all_test_preds = torch.cat(test_preds_list)
            all_test_labels = torch.cat(test_labels_list)
            test_acc = (all_test_preds == all_test_labels).float().mean().item()

            self.evals_result['test_acc'] = test_acc
            self.evals_result['test_loss'].append(test_loss)

            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        self.best_val_loss = best_val_loss

        best_hyperparams = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'window_size': self.window_size,
            'dropout_rate': self.dropout_rate,
            'num_epochs': self.num_epochs,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'early_stopping_patience': self.early_stopping_patience
        }
        self.best_hyperparams = best_hyperparams

        # -----------------------------
        # 【最終学習時のみ】最終モデルの保存
        # -----------------------------
        if fold is None:
            model_save_path = model_save_dir / f"rnn_classifier_{self.target_column}_final.pt"
            print(f"最終モデル保存: {model_save_path}")
            torch.save({
                'state_dict': model.state_dict(),
                'hyperparameters': best_hyperparams
            }, model_save_path)

        return {"val_loss": best_val_loss, "test_loss": test_loss, "test_acc": test_acc}

# =================================
# 以下、foldごとの学習・評価処理など（基本的には変更なし）
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
        target_column=config["target_column"],
        device=device,
        num_classes=config.get("num_classes", 2)
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
    total_loss = 0
    total_acc = 0
    count = 0
    print("\n==== フォールド別結果 ====")
    for res in results:
        fold = res["fold"]
        performance = res["result"]["evaluation_results"]["primary_model"]
        test_loss = performance.get("test_loss", None)
        test_acc = performance.get("accuracy", None)
        print(f"フォールド {fold}: テスト損失 = {test_loss:.4f}, テスト正解率 = {test_acc:.4f}")
        total_loss += test_loss
        total_acc += test_acc
        count += 1
    avg_loss = total_loss / count
    avg_acc = total_acc / count
    print("-----------------------------")
    print(f"全体平均テスト損失: {avg_loss:.4f}")
    print(f"全体平均テスト正解率: {avg_acc:.4f}\n")

def train_model_on_full_data(config: dict, final_hyperparams: dict) -> dict:
    logger.info("全データを用いた最終学習を開始")
    data_dir = config["data_dir"]
    filename = config["filename"]

    # 全データの訓練ファイル
    full_file = data_dir / f"{filename}_train.pkl"
    # テスト用ファイル
    test_file = data_dir / f"{filename}_test.pkl"

    full_df = load_data(full_file)
    test_df = load_data(test_file)

    full_df = full_df.drop(columns=config["exclude_target"])
    test_df = test_df.drop(columns=config["exclude_target"])

    device = config["device"]

    model = RNNBase(
        train_df=full_df,
        test_df=test_df,
        target_column=config["target_column"],
        device=device,
        num_classes=config.get("num_classes", 2)
    )

    # CVで選ばれたハイパーパラメータを上書き
    model.batch_size = final_hyperparams["batch_size"]
    model.learning_rate = final_hyperparams["learning_rate"]
    model.hidden_size = final_hyperparams["hidden_size"]
    model.num_layers = final_hyperparams["num_layers"]
    model.window_size = final_hyperparams["window_size"]
    model.dropout_rate = final_hyperparams["dropout_rate"]
    model.num_epochs = final_hyperparams["num_epochs"]
    model.gradient_accumulation_steps = final_hyperparams["gradient_accumulation_steps"]
    model.early_stopping_patience = final_hyperparams["early_stopping_patience"]

    # 最終学習実行（fold=None で全データを使った再学習）
    final_result_metrics = model.run()  
    final_result_metrics["evals_result"] = model.evals_result

    # テスト結果（予測値・正解値を含む）を保存（pickle形式）
    results_save_dir = Path("storage/predictions")
    results_save_dir.mkdir(parents=True, exist_ok=True)
    results_save_path = results_save_dir / f"rnn_classifier_test_results_{config['target_column']}.pkl"
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

    # CVの結果から、ここではテスト損失が最小のフォールドのハイパーパラメータを採用
    best_fold = min(results, key=lambda x: x["result"]["evaluation_results"]["primary_model"]["test_loss"])
    final_hyperparams = best_fold["best_hyperparams"]
    print("選択された最良ハイパーパラメータ:", final_hyperparams)

    # 全データでの再学習を実施（テストには指定ファイルを使用）
    final_training_result = train_model_on_full_data(config, final_hyperparams)
    print("最終的な全データによる再学習の結果:")
    print(final_training_result)

    return {
        "fold_results": results,
        "final_result": final_training_result
    }

# エントリポイント（デバッグ用）
if __name__ == "__main__":
    config = {
        "num_folds": 5,
        "filename": "bybit_BTCUSDT_15m_data",
        "data_dir": Path("storage/kline"),
        "target_column": "target_label",  # 分類対象のラベル列（整数値であることを前提）
        "exclude_target": ["不要な列1", "不要な列2"],
        "device": "cuda:0",
        "num_classes": 3  # 例：3クラス分類の場合
    }
    results = run_folds(config)
    # ※必要に応じてプロット関数の呼び出しなど
