import sys
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# richによるロギング設定
from rich import print
from rich.logging import RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__name__)

# モデルのインポート（必要なモデルのみ有効にしてください）
#from modules.lightgbm.lightgbm_base import CryptoLightGBM
# from modules.cnn_base.cnn_base import CNNBase
from modules.rnn_base.rnn_base import RNNBase


# ─── ステップ１：データの読み込み ─────────────────────────────
def load_data(file_path: Path) -> pd.DataFrame:
    """
    指定されたパスからpickle形式のデータを読み込み、object型の列があれば警告を出す。
    """
    logger.info(f"データを読み込みます: {file_path}")
    data = pd.read_pickle(file_path)
    logger.info(f"データの読み込み完了: {file_path}")
    print(data.info())
    object_columns = data.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        logger.warning(f"object型の列が存在します: {object_columns}")
    return data


# ─── ステップ２：ファイルパスの生成 ─────────────────────────────
def get_file_paths(data_dir: Path, filename: str, fold: int) -> (Path, Path):
    """
    指定されたディレクトリ・ファイル名・フォールド番号から、学習用とテスト用のファイルパスを生成する。
    """
    train_file = data_dir / f"{filename}_fold{fold}_train.pkl"
    test_file = data_dir / f"{filename}_fold{fold}_test.pkl"
    return train_file, test_file


# ─── ステップ３：各フォールド・モデルの学習 ─────────────────────────────
def train_model_for_fold_and_model(fold: int, model_class, train_file: Path, test_file: Path, target_column: str, device: str):
    """
    指定されたフォールドとモデルクラスに対して、データを読み込み、モデルを初期化して学習を実行する。
    """
    logger.info(f"フォールド {fold} モデル {model_class.__name__} のトレーニングを開始。デバイス: {device}")

    # データ読み込み
    train_df = load_data(train_file)
    test_df = load_data(test_file)

    # モデルの初期化と実行
    model = model_class(
        train_df=train_df,
        test_df=test_df,
        target_column=target_column,
        device=device
    )
    result = model.run(fold)

    logger.info(f"フォールド {fold} モデル {model_class.__name__} の結果: {result}")
    logger.info(f"フォールド {fold} モデル {model_class.__name__} のトレーニング完了\n")
    return result


# ─── ステップ４：結果の集約 ─────────────────────────────
def aggregate_results(results: list):
    """
    各フォールドの学習結果から平均マルチログロスと平均精度を計算する。
    """
    if not results:
        logger.warning("結果がありません。")
        return {}

    avg_multi_logloss = sum(
        res['evaluation_results']['primary_model']['multi_logloss'] for res in results
    ) / len(results)
    avg_accuracy = sum(
        res['evaluation_results']['primary_model']['accuracy'] for res in results
    ) / len(results)

    logger.info(f"全フォールドの平均マルチログロス: {avg_multi_logloss:.4f}")
    logger.info(f"全フォールドの平均精度: {avg_accuracy:.4f}")
    return {"avg_multi_logloss": avg_multi_logloss, "avg_accuracy": avg_accuracy}


# ─── パイプライン全体の実行 ─────────────────────────────
def pipeline(config: dict):
    """
    設定情報（config）に基づいて、全フォールド・全モデルの学習タスクを並列実行し、
    結果を集約する。
    """
    results = []
    tasks = []

    # 並列実行用のスレッドプール（デバイス数に合わせる）
    with ThreadPoolExecutor(max_workers=len(config["available_devices"])) as executor:
        # 各フォールドごとにタスクを生成
        for fold in range(1, config["num_folds"] + 1):
            train_file, test_file = get_file_paths(config["data_dir"], config["filename"], fold)
            # 利用可能なデバイスをラウンドロビンで割り当てる
            device = config["available_devices"][(fold - 1) % len(config["available_devices"])]

            for model_class in config["model_classes"]:
                task = executor.submit(
                    train_model_for_fold_and_model,
                    fold,
                    model_class,
                    train_file,
                    test_file,
                    config["target_column"],
                    device
                )
                tasks.append(task)

        # タスク完了を待ち、結果を収集
        for future in as_completed(tasks):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"タスク実行中にエラー: {e}")

    # 結果の集約
    aggregate_results(results)
    logger.info("全フォールドのトレーニングが完了しました。")
    return results


# ─── エントリポイント ─────────────────────────────
def main():
    # 設定情報の定義
    config = {
        "num_folds": 4,  # 必要に応じてフォールド数を設定
        "available_devices": ["cuda:0","cuda:1"],  # 使用可能なデバイスのリスト（例: ["CPU"], ["cuda:0"] など）
        "symbol": "BTCUSDT",
        "filename": "bybit_BTCUSDT_15m_data",  # ファイル名のベース
        "data_dir": Path("storage/kline"),
        "target_column": "target",
        # 使用するモデルクラスのリスト
        # "model_classes": [CryptoLightGBM, CNNBase, RNNBase],
        "model_classes": [RNNBase],
    }

    # パイプラインの実行
    pipeline(config)


if __name__ == "__main__":
    main()
