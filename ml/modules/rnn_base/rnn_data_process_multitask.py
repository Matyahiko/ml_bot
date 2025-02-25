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
                 target_columns=None,  # 文字列または文字列のリスト（必ず指定してください）
                 window_size: int = 10,
                 stride: int = 30,
                 val_ratio: float = 0.2, 
                 shuffle: bool = False,
                 random_seed: int = 42
                 ):
        """
        コンストラクタ

        Args:
            train_df, val_df, test_df: 学習／検証／テスト用の DataFrame
            predict_df: 推論用 DataFrame（ターゲット列は含まない想定）
            target_columns: ラベル列名（文字列または文字列のリスト）。必ず指定すること。
            window_size: 時系列ウィンドウの長さ
            stride: ウィンドウ作成時のステップ幅
            val_ratio: val_df が指定されない場合、train_df から自動分割する比率
            shuffle: 時系列データなのにシャッフルする場合のオプション（通常は False）
            random_seed: シャッフル用のシード
        """
        # target_columns が文字列の場合はリストに変換
        if target_columns is None:
            raise ValueError("target_columns を必ず指定してください。")
        elif isinstance(target_columns, list):
            self.target_columns = target_columns
        else:
            self.target_columns = [target_columns]

        self.window_size = window_size
        self.stride = stride

        if predict_df is not None:
            # 推論モードでは、ターゲット列があれば除去する
            predict_df = predict_df.drop(columns=self.target_columns, errors='ignore')
            self.mode = 'predict'
            self.predict_df = predict_df.copy()
            print("[bold green]推論モードで初期化しています。[/bold green]")
            
            # 標準化処理はコメントアウト
            # self._standardize_predict_features()
            
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

            # 標準化処理はコメントアウト
            # self._standardize_features()

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

    # 標準化処理（今回は何もしません）
    def _standardize_features(self):
        pass

    def _standardize_predict_features(self):
        pass

    def _convert_to_tensor(self):
        # ターゲット列が指定されている場合、特徴量とラベルを分離して変換
        self.X_train = torch.tensor(
            self.train_df.drop(columns=self.target_columns, errors='ignore').values, 
            dtype=torch.float32
        )
        self.y_train = torch.tensor(
            self.train_df[self.target_columns].values, 
            dtype=torch.float32
        )
        
        if self.val_df is not None:
            self.X_val = torch.tensor(
                self.val_df.drop(columns=self.target_columns, errors='ignore').values, 
                dtype=torch.float32
            )
            self.y_val = torch.tensor(
                self.val_df[self.target_columns].values, 
                dtype=torch.float32
            )
        else:
            self.X_val, self.y_val = None, None
                
        if self.test_df is not None:
            self.X_test = torch.tensor(
                self.test_df.drop(columns=self.target_columns, errors='ignore').values, 
                dtype=torch.float32
            )
            self.y_test = torch.tensor(
                self.test_df[self.target_columns].values, 
                dtype=torch.float32
            )
        else:
            self.X_test, self.y_test = None, None
                
        print('Tensor変換完了:', 
              f"train: {self.X_train.shape}, "
              f"val: {self.X_val.shape if self.X_val is not None else 'None'}, "
              f"test: {self.X_test.shape if self.X_test is not None else 'None'}")

    def _convert_predict_to_tensor(self):
        self.X_predict = torch.tensor(self.predict_df.values, dtype=torch.float32)
        print('推論データのTensor変換完了:', f"predict: {self.X_predict.shape}")

    def _reshape_tensor(self):
        # ヘルパー関数：ウィンドウ作成（特徴量とラベルを別々に扱う）
        def reshape_data(X, y):
            sub_tensors = []
            sub_labels = []
            for i in range(0, len(X) - self.window_size, self.stride):
                window = X[i:i+self.window_size, :]
                label = y[i+self.window_size - 1, :]  # 複数タスクの場合はそのままベクトルとなる
                sub_tensors.append(window)
                sub_labels.append(label)
            if len(sub_tensors) == 0:
                return None, None
            sub_tensors = torch.stack(sub_tensors)
            sub_labels = torch.stack(sub_labels)
            return sub_tensors, sub_labels

        # ターゲット列が指定されている場合のみこちらを使用
        self.train_features, self.train_labels = reshape_data(self.X_train, self.y_train)
        if self.X_val is not None:
            self.val_features, self.val_labels = reshape_data(self.X_val, self.y_val)
        else:
            self.val_features, self.val_labels = None, None
        if self.X_test is not None:
            self.test_features, self.test_labels = reshape_data(self.X_test, self.y_test)
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
