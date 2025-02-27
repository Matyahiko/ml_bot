import os
import sys
import pandas as pd
from joblib import Memory, Parallel, delayed

# 必要なモジュールをインポート
from modules.preprocess.technical_indicators import technical_indicators
from modules.preprocess.triple_barrier_labels import triple_barrier_labels
from modules.preprocess.detect_bos import detect_bos
from modules.preprocess.detect_choch import detect_choch
from modules.preprocess.add_lag_features import add_lag_features
from modules.preprocess.backtest_labeling import backtest_labeling_run

from modules.cross_validation.MovingWindowKFold import MovingWindowKFold

# キャッシュ設定（不要な場合は削除可能）
memory = Memory(location='./joblib_cache/', verbose=0)
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
        
        #データdirを削除して再作成
        os.system("rm -rf storage/kline")
        os.system("mkdir storage/kline")

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

        # 各シンボルに対して特徴量を追加
        for sym in self.symbols:
            df = technical_indicators(df, sym)
            # ここでラグ特徴量を追加（例として1,2,3期間のラグを追加）
            df = add_lag_features(df, sym, lags=[1, 2, 3])
        
        df = backtest_labeling_run(df)
        
        # 欠損値の除去
        df.dropna(inplace=True)
        
        # シャッフル
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df

    # === ステップ3: データの分割＆各フォールドの処理 ===
    def save_processed_data(self, data: pd.DataFrame, fold: int, subset_name: str) -> None:
        """
        加工済みデータを pickle と CSV で保存
        """
        output_path = f"storage/kline/{self.filename}_fold{fold}_{subset_name}.pkl"
        data.to_pickle(output_path)
        csv_path = output_path.replace('.pkl', '.csv')
        data.to_csv(csv_path)

    def process_fold(self, fold: int, train_idx, test_idx):
        train_df = self.dataset.iloc[train_idx]
        test_df = self.dataset.iloc[test_idx]
        
        processed_train = self.preprocess(train_df)
        processed_test = self.preprocess(test_df)

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
    # シンボルなどの設定
    symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
    pipeline = DataPipeline(
        symbols=symbols,
        base_path='raw_data',
        filename='bybit_BTCUSDT_15m',
        num_folds=5,
        n_jobs=-1
    )
    pipeline.run()
