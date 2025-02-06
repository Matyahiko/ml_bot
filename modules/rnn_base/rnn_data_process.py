import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from rich import print

class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 train_df: pd.DataFrame = None, 
                 val_df: pd.DataFrame = None, 
                 test_df: pd.DataFrame = None, 
                 predict_df: pd.DataFrame = None,
                 target_column: str = 'target',
                 window_size: int = 10,
                 stride: int = 100,
                 val_ratio: float = 0.2, 
                 shuffle: bool = False,
                 random_seed: int = 42
                 ):
        """
        コンストラクタ
        Args:
            ・train_df, val_df, test_df: 学習/検証/テスト用のDataFrame
            ・predict_df: 推論用DataFrame（target列なし）
            ・target_column: ラベル列名
            ・window_size: 時系列ウィンドウの長さ
            ・stride: ウィンドウ作成時のステップ幅
            ・val_ratio: val_dfが指定されない場合、train_dfから自動分割する比率
            ・shuffle: 時系列データなのにシャッフルする場合のオプション（通常False）
            ・random_seed: シャッフル用のシード
        """
        
        self.target_column = target_column
        self.window_size = window_size
        self.stride = stride

        if predict_df is not None:
            # 推論モード
            self.mode = 'predict'
            self.predict_df = predict_df.copy()
            print("[bold green]推論モードで初期化しています。[/bold green]")
            self._standardize_predict_features()
            self._convert_predict_to_tensor()
            self._reshape_predict_tensor()
        else:
            # 学習／検証／テストモード
            self.mode = 'train'
            if val_df is None:
                train_df, val_df = self._split_train_val(train_df,
                                                         val_ratio=val_ratio,
                                                         shuffle=shuffle,
                                                         random_seed=random_seed)
            self.train_df = train_df.copy()
            self.val_df = val_df.copy() if val_df is not None else None
            self.test_df = test_df.copy() if test_df is not None else None

            # 標準化：学習データから平均・標準偏差を計算し、各分割に適用
            self._standardize_features()
            self._convert_to_tensor()
            self._reshape_tensor()

    def _split_train_val(self, df, val_ratio=0.2, shuffle=False, random_seed=42):
        if shuffle:
            df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
            print("[bold yellow]【警告】時系列データなのにシャッフルしています。[/bold yellow]")
        
        val_size = int(len(df) * val_ratio)
        train_df = df.iloc[:-val_size].copy()
        val_df = df.iloc[-val_size:].copy()
        
        print(f"時系列分割を行いました。"
              f"トレーニングセット: {len(train_df)} 件, "
              f"バリデーションセット: {len(val_df)} 件")
        
        return train_df, val_df

    def _standardize_features(self):
        if self.target_column in self.train_df.columns:
            train_features = self.train_df.drop(columns=[self.target_column])
        else:
            train_features = self.train_df.iloc[:, :-1]
        
        self.mean_ = train_features.mean()
        self.std_ = train_features.std() + 1e-8  # ゼロ除算防止
        
        if self.target_column in self.train_df.columns:
            self.train_df.loc[:, self.train_df.columns != self.target_column] = (
                self.train_df.loc[:, self.train_df.columns != self.target_column] - self.mean_
            ) / self.std_
        else:
            self.train_df.iloc[:, :-1] = (train_features - self.mean_) / self.std_
        
        # val_dfの標準化
        if self.val_df is not None:
            if self.target_column in self.val_df.columns:
                val_features = self.val_df.drop(columns=[self.target_column])
                self.val_df.loc[:, self.val_df.columns != self.target_column] = (
                    val_features - self.mean_
                ) / self.std_
            else:
                val_features = self.val_df.iloc[:, :-1]
                self.val_df.iloc[:, :-1] = (val_features - self.mean_) / self.std_
        
        # test_df の標準化
        if self.test_df is not None:
            if self.target_column in self.test_df.columns:
                test_features = self.test_df.drop(columns=[self.target_column])
                self.test_df.loc[:, self.test_df.columns != self.target_column] = (
                    test_features - self.mean_
                ) / self.std_
            else:
                test_features = self.test_df.iloc[:, :-1]
                self.test_df.iloc[:, :-1] = (test_features - self.mean_) / self.std_
        
        print('Tensor変換前の標準化完了:', 
              f"train: {self.train_df.shape}, "
              f"val: {self.val_df.shape if self.val_df is not None else 'None'}, "
              f"test: {self.test_df.shape if self.test_df is not None else 'None'}")

    def _standardize_predict_features(self):
        self.mean_ = self.predict_df.mean()
        self.std_ = self.predict_df.std() + 1e-8
        self.predict_df = (self.predict_df - self.mean_) / self.std_
        print('推論データの標準化完了:', f"predict_df: {self.predict_df.shape}")

    def _convert_to_tensor(self):
        self.X_train = torch.tensor(self.train_df.values, dtype=torch.float32)
        
        if self.val_df is not None:
            self.X_val = torch.tensor(self.val_df.values, dtype=torch.float32)
        else:
            self.X_val = None
            
        if self.test_df is not None:
            self.X_test = torch.tensor(self.test_df.values, dtype=torch.float32)
        else:
            self.X_test = None
            
        print('Tensor変換完了:', 
              f"train: {self.X_train.shape}, "
              f"val: {self.X_val.shape if self.X_val is not None else 'None'}, "
              f"test: {self.X_test.shape if self.X_test is not None else 'None'}")

    def _convert_predict_to_tensor(self):
        self.X_predict = torch.tensor(self.predict_df.values, dtype=torch.float32)
        print('推論データのTensor変換完了:', f"predict: {self.X_predict.shape}")

    def _reshape_tensor(self):
        def reshape_data(tensor):
            labels = tensor[:, -1].long()  # ターゲットは最後の列
            features = tensor[:, :-1]
            sub_tensors = []
            sub_labels = []
            for i in range(0, len(tensor) - self.window_size, self.stride):
                sub_tensor = features[i:i + self.window_size, :]
                sub_label = labels[i + self.window_size - 1]  # ※未来予測の場合はオフセット検討
                sub_tensors.append(sub_tensor)
                sub_labels.append(sub_label)
            if len(sub_tensors) == 0:
                return None, None
            sub_tensors = torch.stack(sub_tensors)
            sub_labels = torch.tensor(sub_labels, dtype=torch.long)
            return sub_tensors, sub_labels

        self.train_features, self.train_labels = reshape_data(self.X_train)

        if self.X_val is not None:
            self.val_features, self.val_labels = reshape_data(self.X_val)
        else:
            self.val_features, self.val_labels = None, None

        if self.X_test is not None:
            self.test_features, self.test_labels = reshape_data(self.X_test)
        else:
            self.test_features, self.test_labels = None, None

        print('reshape完了')

    def _reshape_predict_tensor(self):
        sub_tensors = []
        for i in range(0, len(self.X_predict) - self.window_size, self.stride):
            sub_tensor = self.X_predict[i:i + self.window_size, :]
            sub_tensors.append(sub_tensor)
        if len(sub_tensors) == 0:
            self.predict_features = None
        else:
            self.predict_features = torch.stack(sub_tensors)
        print('推論データのreshape完了:', 
              f"predict_features: {self.predict_features.shape if self.predict_features is not None else None}")
    
    # 以下 __len__ と __getitem__ は、主に学習用（mode='train') として利用してください。
    # 検証・テスト用データは get_val_data() / get_test_data() 経由で取得し、TensorDatasetを作成してください。
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_features) if self.train_features is not None else 0
        else:
            return len(self.predict_features) if self.predict_features is not None else 0

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.train_features[idx], self.train_labels[idx]
        else:
            return self.predict_features[idx]
    
    def get_val_data(self):
        return self.val_features, self.val_labels

    def get_test_data(self):
        return self.test_features, self.test_labels

    def get_predict_data(self):
        return self.predict_features