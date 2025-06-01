#1_kline_data_pipeline.py

import os
import sys
import time
from rich import print
import pandas as pd
from joblib import Memory, Parallel, delayed

# 特徴量追加用の関数
from modules.preprocess.technical_indicators import technical_indicators
from modules.preprocess.add_lag_features import add_lag_features
from modules.preprocess.add_time_features import add_time_features 
from modules.preprocess.add_rolling_statistics import add_rolling_statistics
from modules.preprocess.add_divergence import add_divergence

# 特徴量削減用の関数
from modules.preprocess.pca_feature_reduction import pca_feature_reduction

# 教師ラベル追加用の関数
from modules.preprocess.backtest_labeling import backtest_labeling_run
from modules.preprocess.add_price_change_rate import add_price_change_rate
from modules.preprocess.add_future_volatility import add_future_volatility
from modules.preprocess.add_excess_return import add_excess_return
from modules.preprocess.add_price_direction_label import add_price_direction_label

# 時系列クロスバリデーション用のクラス
from modules.cross_validation.MovingWindowKFold import MovingWindowKFold

# 正規化用の関数をインポート
from modules.preprocess.normalization import fit_transform_scaler, transform_scaler

os.chdir("/app/ml_bot/ml")


class DataPipeline:
    def __init__(self, symbols, base_path='raw_data', filename='bybit_BTCUSDT_15m', num_folds=5, n_jobs=-1):
        self.symbols = symbols
        self.base_path = base_path
        self.filename = filename
        self.num_folds = num_folds
        self.n_jobs = n_jobs
        self.dataset = None
        
    # === ステップ1: データの読み込みと統合 ===
    def load_and_merge_data(self):
    
        main_df = None
        for sym in self.symbols:
            file_path = os.path.join(self.base_path, f'bybit_{sym}_15m.csv')
            df = pd.read_csv(file_path)
            # CSVの'timestamp'列をUTCとして読み込み、その後timezone情報を除去してtimezone-naiveに変換
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
            # 'timestamp'をインデックスに設定し、時系列順にソート
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            # シンボルごとにカラム名にプレフィックスを追加
            df = df.add_prefix(f'{sym}_')
            # 統合（内部結合）
            if main_df is None:
                main_df = df
            else:
                main_df = main_df.join(df, how='inner')
        # 後続処理のためにインデックスを元のカラムに戻す
        self.dataset = main_df.reset_index()

    # === ステップ2: 前処理 ===
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # 時間ベースの特徴量を追加
        df = add_time_features(df)

        # 各シンボルに対して特徴量を追加
        for sym in self.symbols:
            # テクニカル指標を追加
            df = technical_indicators(df, sym)
            # 指標間の差を追加
            # df = add_divergence(df, sym)
            # ローリング統計量を追加
            df = add_rolling_statistics(df, sym, windows=[5, 10, 20])
            # ラグ特徴量を追加
            df = add_lag_features(df, sym, lags=[1, 2, 3])
        
        # 目的変数: ログリターン、ボラティリティ、最大ドローダウンを追加
        df = backtest_labeling_run(df)
        
        # 目的変数: 価格変化方向ラベルを追加
        # df = add_price_direction_label(df, self.symbols[0], n=1)

        # 目的変数: nステップ先の価格変化率を追加
        # df = add_price_change_rate(df, self.symbols[1], n=1)

        # 目的変数: nステップ先のボラティリティを追加（最低でもn=2以上でないと計算できない）
        # df = add_future_volatility(df, self.symbols[0], n=2)

        # 目的変数: nステップ先の超過リターンを追加（benchmark_close カラムが存在する前提）
        # df = add_excess_return(df, self.symbols, n=1, benchmark_col="benchmark_close")
        
        # 欠損値の除去
        df.dropna(inplace=True)
        
        # 既に前処理済みのDataFrame df に対して、全数値型カラムにPCAを適用し、99%の累積説明分散比率を保持する
        # df = pca_feature_reduction(df, n_components=0.99)
       
        # 時間的な依存関係は各行におさまっているので
        df = df.sample(frac=1).reset_index(drop=True)
       
        return df

    # === ステップ3: データの分割＆各フォールドの処理 ===
    def save_processed_data(self, data: pd.DataFrame, fold: int, subset_name: str) -> None:
        """
        加工済みデータを pickle と CSV で保存
        """
        output_path = f"storage/kline/{self.filename}_fold{fold}_{subset_name}.pkl"
        data.to_pickle(output_path)
        # 遅いのでコメントアウト
        # csv_path = output_path.replace('.pkl', '.csv')
        # data.to_csv(csv_path)

    def process_fold(self, fold: int, train_idx, test_idx):
        train_df = self.dataset.iloc[train_idx]
        test_df = self.dataset.iloc[test_idx]
        
        processed_train = self.preprocess(train_df)
        processed_test = self.preprocess(test_df)
        
        # --- 正規化の適用 ---
        norm_method = 'minmax'
        
        # 目的変数のカラム名（必要に応じて変更してください）
        target_columns = ['log_return', 'volatility', 'max_drawdown']
        
        # 特徴量と目的変数に分離（目的変数が存在しない場合はエラーにならないように）
        X_train = processed_train.drop(columns=target_columns, errors='ignore')
        y_train = processed_train[target_columns] if set(target_columns).issubset(processed_train.columns) else pd.DataFrame(index=processed_train.index)
        
        X_test = processed_test.drop(columns=target_columns, errors='ignore')
        y_test = processed_test[target_columns] if set(target_columns).issubset(processed_test.columns) else pd.DataFrame(index=processed_test.index)
        
        # 特徴量のみを正規化
        X_train, scaler_obj = fit_transform_scaler(X_train, method=norm_method)
        X_test = transform_scaler(X_test, scaler_obj)
        
        # 再結合
        processed_train = pd.concat([X_train, y_train], axis=1)
        processed_test = pd.concat([X_test, y_test], axis=1)
        
        import joblib
        joblib.dump(scaler_obj, 'storage/kline/minmax_scaler.joblib')
        
        self.save_processed_data(processed_train, fold, 'train')
        self.save_processed_data(processed_test, fold, 'test')

    def split_and_process(self):
        # 分割前に index をリセット
        dataset = self.dataset.reset_index(drop=True)
        tscv = MovingWindowKFold(ts_column='timestamp', n_splits=self.num_folds, clipping=False)
        folds = list(tscv.split(dataset))

        Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_fold)(fold, train_idx, test_idx)
            for fold, (train_idx, test_idx) in enumerate(folds)
        )

    # === パイプライン全体の実行 ===
    def run(self):
        print("Step 1: Loading and merging data...")
        self.load_and_merge_data()

        print("Step 2: Splitting data and processing each fold...")
        self.split_and_process()

        print("Data processing pipeline complete.")

if __name__ == '__main__':
    
        # データdirを削除して再作成
    os.system("rm -rf storage/kline")
    os.system("mkdir storage/kline")
    os.system("mkdir storage/kline/temp")
    
    # シンボルなどの設定
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    pipeline = DataPipeline(
        symbols=symbols,
        base_path='raw_data',
        filename='bybit_BTCUSDT_15m',
        num_folds=5,
        n_jobs=-1
    )
    start_time = time.time()
    pipeline.run()
    elapsed_time = time.time() - start_time
    
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"\nTotal execution time: {hours}時間 {minutes}分 {seconds:.2f}秒")
