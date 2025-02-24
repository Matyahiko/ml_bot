import os
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # ROC-AUC用にsoftmaxなどを使う場合
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# 別ファイルの可視化用モジュールをインポート
from modules.cnn_base.plot_utils import plot_all_results
# 改良した VisionTransformer をインポート
from modules.cnn_base.cnn_base_arc import ModernBayesianCNN
from modules.cnn_base.cnn_data_process import ConvertDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    MODEL_DIR = Path('models/nn/')
    PREDICT_DIR = Path('storage/predictions/')
    STORAGE_DIR = Path('storage/cnn_base/')

    @staticmethod
    def ensure_directories():
        os.makedirs(Config.MODEL_DIR, exist_ok=True)
        os.makedirs(Config.PREDICT_DIR, exist_ok=True)
        os.makedirs(Config.STORAGE_DIR, exist_ok=True)


class CNNBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.train_df = kwargs.get('train_df')
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df')
        self.target_column = kwargs.get('target_column')
        self.device = kwargs.get('device', 'cuda:0')

        # デフォルトハイパーパラメータ
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.embed_dim = 512
        self.depth = 6
        self.num_heads = 8
        self.mlp_ratio = 1024
        self.dropout = 0.1
        self.num_epochs = 10

        # ハイパーパラ探索を行うかどうかのフラグ
        self.do_hyperparam_search = False

        Config.ensure_directories()

        self.data_set = ConvertDataset(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            target_column=self.target_column
        )

        # 可視化用に損失・精度、予測結果などを格納する辞書
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
        Parameters
        ----------
        fold : int, optional
            Fold番号（交差検証などを行う場合用）
        """

        # もしハイパーパラ探索を行うならobjective関数などを作って最適化
        if self.do_hyperparam_search:
            def objective(trial: Trial):
                # ハイパーパラサンプリング
                batch_size = trial.suggest_categorical('batch_size', [16, 32])
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
                num_heads = trial.suggest_int("num_heads", 1, 16)
                embed_dim = trial.suggest_int("embed_dim", num_heads * 16, num_heads * 32, step=num_heads)
                depth = trial.suggest_int('depth', 4, 12)
                mlp_ratio = trial.suggest_categorical('mlp_ratio', [512, 1024, 2048])
                dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
                num_epochs = 2

                # 元の値を退避
                orig_batch_size = self.batch_size
                orig_learning_rate = self.learning_rate
                orig_embed_dim = self.embed_dim
                orig_depth = self.depth
                orig_num_heads = self.num_heads
                orig_mlp_ratio = self. mlp_ratio
                orig_dropout = self.dropout
                orig_num_epochs = self.num_epochs

                # 試行パラメータで上書き
                self.batch_size = batch_size
                self.learning_rate = learning_rate
                self.embed_dim = embed_dim
                self.depth = depth
                self.num_heads = num_heads
                self. mlp_ratio =  mlp_ratio
                self.dropout = dropout
                self.num_epochs = num_epochs

                val_loss = self._train_and_evaluate()

                # 元の値に戻す
                self.batch_size = orig_batch_size
                self.learning_rate = orig_learning_rate
                self.embed_dim = orig_embed_dim
                self.depth = orig_depth
                self.num_heads = orig_num_heads
                self. mlp_ratio = orig_mlp_ratio
                self.dropout = orig_dropout
                self.num_epochs = orig_num_epochs

                return val_loss

            # Optunaでハイパーパラ探索
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=5)
            )
            study.optimize(objective, n_trials=5)

            # 最良パラメータを適用（再学習したい場合など）
            best_params = study.best_trial.params
            self.batch_size = best_params['batch_size']
            self.learning_rate = best_params['learning_rate']
            self.embed_dim = best_params['embed_dim']
            self.depth = best_params['depth']
            self.num_heads = best_params['num_heads']
            self. mlp_ratio = best_params['mlp_ratio']
            self.dropout = best_params['dropout']
            # チューニング後の再学習を行うならここでエポックを調整
            self.num_epochs = 2

        # 最終的に学習・評価
        self._train_and_evaluate(fold=fold)

        # === ここで可視化用スクリプトを呼び出す ===
        #    データフレームのtarget_columnに応じてクラス名を決めてください
        class_names = [0, 1]

        plot_all_results(
            evals_result=self.evals_result,
            class_names=class_names
        )
        # ==============================

    def _train_and_evaluate(self, fold=None):
        batch_size = self.batch_size
        learning_rate = self.learning_rate
        num_epochs = self.num_epochs

        # データローダ (例)
        train_loader = DataLoader(self.data_set, batch_size=batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(self.data_set, batch_size=batch_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(self.data_set, batch_size=batch_size, shuffle=False, num_workers=1)

        # Bayesianモデル
        model = ModernBayesianCNN(num_classes=2)

        # 損失とオプティマイザ
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

        device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        criterion = criterion.to(device)

        best_val_loss = float('inf')
        best_model_wts = copy.deepcopy(model.state_dict())

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)

            # -------------------
            # 1) トレーニング
            # -------------------
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in tqdm(train_loader, desc="Training", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # ← [重要] Bayesianモデルは (logits, kl) のタプルを返す
                outputs, kl = model(inputs)
                
                # 予測ラベルを取得
                _, preds = torch.max(outputs, 1)

                # CrossEntropy (CE) 損失
                ce_loss = criterion(outputs, labels)

                # KLを何倍にするか: β
                beta = 1e-3
                loss = ce_loss + beta * kl

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            self.evals_result['train_loss'].append(epoch_loss)
            self.evals_result['train_acc'].append(epoch_acc.item())
            print(f"Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # -------------------
            # 2) バリデーション
            # -------------------
            if val_loader is not None:
                model.eval()
                val_running_loss = 0.0
                val_running_corrects = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # 同様に (outputs, kl) のタプル
                        outputs, kl = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        # バリデーション時は CE のみを評価する例
                        loss = criterion(outputs, labels)

                        val_running_loss += loss.item() * inputs.size(0)
                        val_running_corrects += torch.sum(preds == labels.data)
                        val_total += labels.size(0)

                val_epoch_loss = val_running_loss / val_total
                val_epoch_acc = val_running_corrects.double() / val_total

                self.evals_result['val_loss'].append(val_epoch_loss)
                self.evals_result['val_acc'].append(val_epoch_acc.item())
                print(f"Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")

                # ベストモデルを更新
                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("Best model updated.")
            print()

        # ベストモデルをロード
        model.load_state_dict(best_model_wts)

        # -------------------
        # 3) テスト
        # -------------------
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

                    outputs, kl = model(inputs)  # タプル
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
            test_acc = test_running_corrects.double() / test_total
            print(f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}")

            self.evals_result['test_labels'] = all_labels
            self.evals_result['test_preds'] = all_preds
            self.evals_result['test_probas'] = all_probas

        print("Training complete.")
        self.best_val_loss = best_val_loss
        print(f"Best validation loss: {best_val_loss:.4f}")
        return best_val_loss