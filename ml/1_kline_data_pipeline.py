import os
import sys
import pandas as pd
from joblib import Memory, Parallel, delayed

# 必要なモジュールをインポート
from modules.preprocess.technical_indicators import technical_indicators
from modules.preprocess.triple_barrier_labels import triple_barrier_labels
from modules.preprocess.trend_scan_labels import trend_scan_labels
from modules.preprocess.detect_bos import detect_bos
from modules.preprocess.detect_choch import detect_choch
from modules.preprocess.fvg import detect_fvg, fvg_regression
from modules.preprocess.wvf import wvf
from modules.preprocess.wvf_triple import wvf_triple
from modules.cross_validation.MovingWindowKFold import MovingWindowKFold

# キャッシュ設定（不要な場合は削除可能）
memory = Memory(location='./joblib_cache/', verbose=0)

class DataPipeline:
    """
    データ処理のパイプラインをクラスとして実装
      1. データの読み込みと統合
      2. 前処理（特徴量作成、ラベル作成、欠損値処理など）
      3. 時系列データの分割とフォールドごとの処理・保存
    """
    def __init__(self, symbols, base_path='raw_data', filename='bybit_BTCUSDT_15m_data', num_folds=5, n_jobs=-1):
        self.symbols = symbols
        self.base_path = base_path
        self.filename = filename
        self.num_folds = num_folds
        self.n_jobs = n_jobs
        self.dataset = None

    # === ステップ1: データの読み込みと統合 ===
    def load_and_merge_data(self):
        """
        各シンボルのCSVファイルを読み込み、時系列で統合する
        ※ 各シンボルのカラム名にシンボル名のプレフィックスを付与
        """
        main_df = None
        for sym in self.symbols:
            file_path = os.path.join(self.base_path, f'bybit_{sym}_15m_data.csv')
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
        """
        前処理内容：
          - タイムスタンプを index にセットし、ソート
          - 各シンボルの特徴量を追加
          - ターゲットラベルの作成（例：trend_scan_labels）
          - 欠損値の除去
        """
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

        # 各シンボルに対して特徴量を追加
        for sym in self.symbols:
            df = technical_indicators(df, sym)

        # 最初のシンボルを用いてラベル作成（必要に応じて他のラベル作成関数も利用可能）
        df = trend_scan_labels(df, self.symbols[0])

        # 必要に応じた処理例：undersampling など（コメントアウト）
        # df = undersample_majority_class(df, 'target')

        # 欠損値の除去
        df.dropna(inplace=True)
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
        """
        単一フォールドの train/test データに対して前処理を実行し、保存する
        """
        train_df = self.dataset.iloc[train_idx]
        test_df = self.dataset.iloc[test_idx]
        
        processed_train = self.preprocess(train_df)
        processed_test = self.preprocess(test_df)

        self.save_processed_data(processed_train, fold, 'train')
        self.save_processed_data(processed_test, fold, 'test')

    def split_and_process(self):
        """
        時系列KFoldでデータを分割し、各フォールドごとに並列処理を実行する
        """
        # 分割前に index をリセット
        dataset = self.dataset.reset_index(drop=True)
        tscv = MovingWindowKFold(ts_column='timestamp', n_splits=self.num_folds, clipping=True)
        folds = list(tscv.split(dataset))

        Parallel(n_jobs=self.n_jobs)(
            delayed(self.process_fold)(fold, train_idx, test_idx)
            for fold, (train_idx, test_idx) in enumerate(folds)
        )

    # === パイプライン全体の実行 ===
    def run(self):
        """
        パイプライン全体の流れを実行
          1. データ読み込みと統合
          2. 分割＆各フォールドの前処理・保存
        """
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
        filename='bybit_BTCUSDT_15m_data',
        num_folds=5,
        n_jobs=-1
    )
    pipeline.run()
