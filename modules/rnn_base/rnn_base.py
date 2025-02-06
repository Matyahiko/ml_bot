import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path

# Optuna関連
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import Trial

# ユーザー定義の前処理済みデータセットなど
# ※TimeSeriesDataset側でwindow_size引数を受け取るように実装しておくこと
from modules.rnn_base.rnn_data_process import TimeSeriesDataset
from modules.cnn_base.plot_utils import plot_all_results

# =================================
# シンプルなRNNモデル例（GRUベースの二値分類、ドロップアウト追加）
# =================================
class SimpleGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=1, num_classes=2, dropout_rate=0.0):
        """
        dropout_rate: GRU後のドロップアウト率（0.0～0.5程度を想定）
        """
        super().__init__()
        # 複数層の場合、nn.GRUのdropout引数を設定（1層の場合は無視される）
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        # GRU出力後にドロップアウトを適用
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # 最終タイムステップの出力を取り出す
        out_last = out[:, -1, :]  # (batch_size, hidden_size)
        out_last = self.dropout(out_last)
        logits = self.fc(out_last)  # (batch_size, num_classes)
        return logits

# =================================
# RNNBase クラス（学習・検証・テストのDataLoader作成を分離）
# =================================
class RNNBase(nn.Module):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        train_df : pd.DataFrame
        val_df : pd.DataFrame
        test_df : pd.DataFrame
        target_column : str
            ラベル列の名前
        device : str
            "cuda:0" or "cpu" etc.
        その他ハイパーパラメータはkwargsで取得
        """
        super().__init__()
        self.train_df = kwargs.get('train_df')
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df')
        self.target_column = kwargs.get('target_column')
        self.device = kwargs.get('device', 'cuda:0')

        # デフォルトのハイパーパラメータ
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 10
        self.hidden_size = 128
        self.num_layers = 1
        self.window_size = 50  # デフォルトのウィンドウサイズ
        self.dropout_rate = 0.0  # デフォルトはドロップアウトなし

        # ハイパーパラ探索の有無
        self.do_hyperparam_search = True

        # 前処理済みデータセットの作成（初期値はデフォルトのwindow_sizeで作成）
        self.data_set = TimeSeriesDataset(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            target_column=self.target_column,
            window_size=self.window_size
        )

        # 可視化用：損失・精度などの記録
        self.evals_result = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_labels': [],
            'test_preds': [],
            'test_probas': []
        }

        self.best_val_loss = float('inf')

    def run(self, fold: int = None):
        """
        メイン実行メソッド
        """
        if self.do_hyperparam_search:
            def objective(trial: Trial):
                # ハイパーパラメータのサンプリング例
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
                hidden_size = trial.suggest_int('hidden_size', 32, 256, step=32)
                num_layers = trial.suggest_int('num_layers', 1, 3)
                window_size = trial.suggest_int('window_size', 10, 100, step=10)
                dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
                num_epochs = 3  # 試行ごとに短縮

                # 元の値を保存
                orig_batch_size = self.batch_size
                orig_lr = self.learning_rate
                orig_hidden_size = self.hidden_size
                orig_num_layers = self.num_layers
                orig_num_epochs = self.num_epochs
                orig_window_size = self.window_size
                orig_dropout_rate = self.dropout_rate

                # 上書き
                self.batch_size = batch_size
                self.learning_rate = learning_rate
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.num_epochs = num_epochs
                self.window_size = window_size
                self.dropout_rate = dropout_rate

                # datasetの再生成（window_sizeを含むパラメータが変更されるため）
                self.data_set = TimeSeriesDataset(
                    train_df=self.train_df,
                    val_df=self.val_df,
                    test_df=self.test_df,
                    target_column=self.target_column,
                    window_size=self.window_size
                )

                # ※検証データの損失で評価（テストデータは最終評価のみ）
                result = self._train_and_evaluate()
                val_loss = result.get("val_loss", float('inf'))

                # 元に戻す
                self.batch_size = orig_batch_size
                self.learning_rate = orig_lr
                self.hidden_size = orig_hidden_size
                self.num_layers = orig_num_layers
                self.num_epochs = orig_num_epochs
                self.window_size = orig_window_size
                self.dropout_rate = orig_dropout_rate

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
            self.num_epochs = 5  # 最終学習時はエポック数を増やすなど適宜調整

            # 最終学習前にdatasetを再生成
            self.data_set = TimeSeriesDataset(
                train_df=self.train_df,
                val_df=self.val_df,
                test_df=self.test_df,
                target_column=self.target_column,
                window_size=self.window_size
            )

            print("Best hyperparameters found:", best_params)

        # 最終学習＆評価
        result_metrics = self._train_and_evaluate(fold=fold)

        # 可視化（2クラス分類想定）
        class_names = [0, 1]
        plot_all_results(
            evals_result=self.evals_result,
            class_names=class_names
        )

        return {
            'evaluation_results': {
                'primary_model': {
                    'multi_logloss': result_metrics.get("test_loss", None),
                    'accuracy': result_metrics.get("test_acc", None)
                }
            }
        }

    def _train_and_evaluate(self, fold=None):
        """
        学習・検証・テストを実施。検証・テストのデータは、
        TimeSeriesDatasetで前処理された各テンソルからTensorDatasetを作成して利用。
        """
        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')

        # --- 各分割ごとにTensorDatasetを作成 ---
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

        # --- モデル初期化 ---
        sample_x, _ = next(iter(train_loader))
        feature_dim = sample_x.shape[-1]

        model = SimpleGRUModel(
            input_size=feature_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=2,
            dropout_rate=self.dropout_rate
        )
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        # 重み減衰（weight_decay）は過学習防止のために利用。必要に応じて調整してください
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())

        # --- 学習ループ ---
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print("-" * 10)

            # トレーニング
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            self.evals_result['train_loss'].append(epoch_loss)
            self.evals_result['train_acc'].append(epoch_acc.item())
            print(f"Training Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            # 検証
            if val_loader is not None:
                model.eval()
                val_running_loss = 0.0
                val_running_corrects = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        val_running_loss += loss.item() * inputs.size(0)
                        val_running_corrects += torch.sum(preds == labels.data)
                        val_total += labels.size(0)

                val_epoch_loss = val_running_loss / val_total
                val_epoch_acc = val_running_corrects.double() / val_total

                self.evals_result['val_loss'].append(val_epoch_loss)
                self.evals_result['val_acc'].append(val_epoch_acc.item())
                print(f"Validation Loss: {val_epoch_loss:.4f}  Acc: {val_epoch_acc:.4f}")

                # ベストモデル更新
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("Best model updated.")
            print()

        # ベストモデルをロード
        model.load_state_dict(best_model_wts)

        # --- テスト評価 ---
        test_loss = None
        test_acc = None
        if test_loader is not None:
            model.eval()
            test_running_loss = 0.0
            test_running_corrects = 0
            test_total = 0

            all_labels = []
            all_preds = []
            all_probas = []

            with torch.no_grad():
                for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    probas = F.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)

                    loss = criterion(outputs, labels)

                    test_running_loss += loss.item() * inputs.size(0)
                    test_running_corrects += torch.sum(preds == labels.data)
                    test_total += labels.size(0)

                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probas.extend(probas.cpu().numpy())

            test_loss = test_running_loss / test_total
            test_acc = (test_running_corrects.double() / test_total).item()
            print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

            self.evals_result['test_labels'] = all_labels
            self.evals_result['test_preds'] = all_preds
            self.evals_result['test_probas'] = all_probas

        print("Training complete.")
        self.best_val_loss = best_val_loss
        print(f"Best validation loss: {best_val_loss:.4f}")

        # --- モデル保存（学習に用いたハイパーパラメータも一緒に保存） ---
        model_save_dir = Path("models/rnn_base")
        model_save_dir.mkdir(parents=True, exist_ok=True)
        if fold is not None:
            model_save_path = model_save_dir / f"rnn_model_fold{fold}.pt"
        else:
            model_save_path = model_save_dir / "rnn_model.pt"

        # 学習に用いたハイパーパラメータをまとめる
        best_hyperparams = {
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'window_size': self.window_size,
            'dropout_rate': self.dropout_rate,
            'num_epochs': self.num_epochs
        }

        # モデルのstate_dictとハイパーパラメータを1ファイルに保存
        torch.save({
            'state_dict': model.state_dict(),
            'hyperparameters': best_hyperparams
        }, model_save_path)
        print(f"Best model and hyperparameters saved to {model_save_path}")

        return {"val_loss": best_val_loss, "test_loss": test_loss, "test_acc": test_acc}