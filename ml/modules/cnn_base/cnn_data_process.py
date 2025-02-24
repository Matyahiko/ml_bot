import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

from torch.utils.data import Dataset, DataLoader, TensorDataset
from rich import print
import logging
from rich.logging import RichHandler
from rich.table import Table

# 画像保存用
import matplotlib.pyplot as plt


logging.basicConfig(
   level="NOTSET",
   format="%(message)s",
   datefmt="[%X]",
   handlers=[RichHandler()],
)

class ConvertDataset(Dataset):
    def __init__(self, 
                 train_df: pd.DataFrame, 
                 val_df: pd.DataFrame = None, 
                 test_df: pd.DataFrame = None, 
                 target_column: str = 'target'):
        
        self.train_df = train_df.copy()
        self.val_df = val_df.copy() if val_df is not None else None
        self.test_df = test_df.copy() if test_df is not None else None
        self.target_column = target_column
        
        # 特徴量の数と合わせると正方形で扱いやすいので、window_sizeを405に設定
        self.window_size = 405
        self.stride = 1
        
        # ---- ① 標準化の事前処理を行う関数を実行 ----
        self._standardize_features()
        
        # ---- ② Torch Tensor化 → ③ reshape処理 → ④ バリデーションデータ分割 ----
        self._convert_to_tensor()
        self._reshape_tensor()
        self._separete_val_data()
        #self.summary()
        #self.save_sample_images()
        
    def _standardize_features(self):
        """
        学習データから平均・標準偏差を算出し、それを用いて全データ(train/val/test)の
        特徴量を標準化する関数。
        """
        # 学習データから特徴量を抽出（最終列がラベル）
        train_features = self.train_df.iloc[:, :-1]
        
        # 学習データの平均・標準偏差を計算 (列方向)
        self.mean_ = train_features.mean()
        self.std_ = train_features.std() + 1e-8  # 標準偏差が0にならないようepsilonを加える
        
        # 学習データを標準化
        self.train_df.iloc[:, :-1] = (train_features - self.mean_) / self.std_
        
        # バリデーションデータがあれば標準化
        if self.val_df is not None:
            val_features = self.val_df.iloc[:, :-1]
            self.val_df.iloc[:, :-1] = (val_features - self.mean_) / self.std_
        
        # テストデータがあれば標準化
        if self.test_df is not None:
            test_features = self.test_df.iloc[:, :-1]
            self.test_df.iloc[:, :-1] = (test_features - self.mean_) / self.std_

    def _convert_to_tensor(self):
        # train_dfをtorch.Tensorに変換
        self.X_train = torch.tensor(self.train_df.values, dtype=torch.float32)
        print(self.X_train.shape)
        
        # val_dfをtorch.Tensorに変換
        if self.val_df is not None:
            self.X_val = torch.tensor(self.val_df.values, dtype=torch.float32)
            
        
        # test_dfをtorch.Tensorに変換
        if self.test_df is not None:
            self.X_test = torch.tensor(self.test_df.values, dtype=torch.float32)
            

        print('Tensor変換完了')
    
    def _reshape_tensor(self):
        def reshape_data(tensor):
            labels = tensor[:, -1].long()  # ターゲットをLong型に変換
            features = tensor[:, :-1]

            sub_tensors = []
            sub_labels = []

            for i in range(0, len(tensor) - self.window_size, self.stride):
                sub_tensor = features[i:i + self.window_size, :]  # (window_size, 特徴量数)
                sub_label = labels[i + self.window_size - 1]      # 予測対象のラベル
                sub_tensors.append(sub_tensor)
                sub_labels.append(sub_label)

            sub_tensors = torch.stack(sub_tensors)               # (サンプル数, window_size, 特徴量数)
            sub_labels = torch.tensor(sub_labels, dtype=torch.long)

            # チャンネル次元を追加: [サンプル数, 1, window_size, 特徴量数]
            sub_tensors = sub_tensors.unsqueeze(1)

            return sub_tensors, sub_labels

        # Train
        self.train_features, self.train_labels = reshape_data(self.X_train)

        # Val
        if self.val_df is not None:
            self.val_features, self.val_labels = reshape_data(self.X_val)

        # Test
        if self.test_df is not None:
            self.test_features, self.test_labels = reshape_data(self.X_test)

        print('reshape完了')
    
    def _separete_val_data(self, val_ratio=0.2, shuffle=True, random_seed=42):
        """
        バリデーションデータが未指定の場合のみ分割する。
        """
        if self.val_df is not None:
            print("バリデーションデータが既に提供されています。分割をスキップします。")
            return

        if shuffle:
            self.train_df = self.train_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            print("トレーニングデータをシャッフルしました。")

        val_size = int(len(self.train_df) * val_ratio)
        self.val_df = self.train_df.iloc[:val_size].copy()
        self.train_df = self.train_df.iloc[val_size:].copy()

        print(f"トレーニングデータを分割しました。"
              f"トレーニングセット: {len(self.train_df)} 件, "
              f"バリデーションセット: {len(self.val_df)} 件")
    
    # 以下のメソッドはDataloaderのために必要
    def __len__(self):
        return len(self.train_features)

    def __getitem__(self, idx):
        return self.train_features[idx], self.train_labels[idx]
    
    # バリデーション・テスト用データを取得
    def get_val_data(self):
        return self.val_features, self.val_labels

    def get_test_data(self):
        return self.test_features, self.test_labels

    def summary(self):
        """
        全データセットの情報をまとめて可視化する。
        テンソルの形状やラベル分布などが確認できる。
        """
        print("[bold magenta]\n=== Dataset Summary ===[/bold magenta]")
        
        # mean_, std_ がある場合は表示
        if hasattr(self, 'mean_') and hasattr(self, 'std_'):
            print("\n[bold cyan]Standardization Parameters[/bold cyan]")
            table_std = Table(show_header=True, header_style="bold green")
            table_std.add_column("Feature")
            table_std.add_column("Mean", justify="right")
            table_std.add_column("Std", justify="right")

            for col in self.mean_.index:
                table_std.add_row(str(col), f"{self.mean_[col]:.4f}", f"{self.std_[col]:.4f}")
            print(table_std)

        # 学習データ
        print("\n[bold cyan]Train Data[/bold cyan]")
        table_train = Table(show_header=True, header_style="bold green")
        table_train.add_column("Item")
        table_train.add_column("Value", justify="right")

        # サブテンソル形状
        table_train.add_row("Features shape", str(tuple(self.train_features.shape)))
        # ラベル形状
        table_train.add_row("Labels shape", str(tuple(self.train_labels.shape)))

        # ラベル分布
        label_counts_train = pd.Series(self.train_labels.numpy()).value_counts().sort_index()
        label_dist_str = ", ".join([f"{int(lbl)}: {cnt}" for lbl, cnt in label_counts_train.items()])
        table_train.add_row("Label distribution", label_dist_str)

        print(table_train)

        # バリデーションデータ
        if hasattr(self, 'val_features') and self.val_features is not None:
            print("\n[bold cyan]Validation Data[/bold cyan]")
            table_val = Table(show_header=True, header_style="bold green")
            table_val.add_column("Item")
            table_val.add_column("Value", justify="right")

            table_val.add_row("Features shape", str(tuple(self.val_features.shape)))
            table_val.add_row("Labels shape", str(tuple(self.val_labels.shape)))

            label_counts_val = pd.Series(self.val_labels.numpy()).value_counts().sort_index()
            label_dist_str_val = ", ".join([f"{int(lbl)}: {cnt}" for lbl, cnt in label_counts_val.items()])
            table_val.add_row("Label distribution", label_dist_str_val)

            print(table_val)

        # テストデータ
        if hasattr(self, 'test_features') and self.test_features is not None:
            print("\n[bold cyan]Test Data[/bold cyan]")
            table_test = Table(show_header=True, header_style="bold green")
            table_test.add_column("Item")
            table_test.add_column("Value", justify="right")

            table_test.add_row("Features shape", str(tuple(self.test_features.shape)))
            table_test.add_row("Labels shape", str(tuple(self.test_labels.shape)))

            label_counts_test = pd.Series(self.test_labels.numpy()).value_counts().sort_index()
            label_dist_str_test = ", ".join([f"{int(lbl)}: {cnt}" for lbl, cnt in label_counts_test.items()])
            table_test.add_row("Label distribution", label_dist_str_test)

            print(table_test)

        print("[bold magenta]=== End of Summary ===[/bold magenta]\n")
    
    # ここから追加：サンプルデータを画像で保存する関数
    def save_sample_images(self, num_samples=5, save_dir='plots/cnn_base/'):
        """
        学習データの最初の num_samples 件を画像として保存する。
        画像化するテンソル形状: (1, window_size, 特徴量数) -> (window_size, 特徴量数) のグレースケール画像。
        """
        # 保存先のディレクトリを作成
        os.makedirs(save_dir, exist_ok=True)

        # 学習データが何サンプルあるか
        total_samples = len(self.train_features)

        # 保存するサンプル数を、データセット数と比較して最小化
        num_samples = min(num_samples, total_samples)

        for i in range(num_samples):
            # shape: (1, window_size, feature_size)
            img_tensor = self.train_features[i]
            # チャンネル次元を落として (window_size, feature_size)
            img_array = img_tensor.squeeze(0).numpy()

            # プロットして画像として保存
            plt.figure(figsize=(6, 6))
            plt.imshow(img_array, cmap='gray', aspect='auto')  
            plt.colorbar() 
            plt.title(f"Sample {i} - Label: {int(self.train_labels[i].item())}")
            plt.xlabel("Feature dimension")
            plt.ylabel("Time / Window axis")

            save_path = os.path.join(save_dir, f"train_sample_{i}.png")
            plt.savefig(save_path)
            plt.close()

        print(f"\n[bold green]{num_samples} 件のサンプル画像を '{save_dir}' フォルダに保存しました。[/bold green]")
