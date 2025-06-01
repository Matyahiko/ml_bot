import time
from rich import print
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# 必要な run_folds 関数のインポート
from modules.rnn_base.rnn_base_regression import run_folds as run_folds_regression
from modules.rnn_base.rnn_base_classification import run_folds as run_folds_classification
from modules.rnn_base.rnn_base_multitask import run_folds as run_folds_multitask
from modules.catboost.run import run as run_catboost

os.chdir("/app/ml_bot/ml")

# モデルごとに全設定をまとめる
MODEL_CONFIGS = {
    "multi_target_model": {
        "num_folds": 3,
        "filename": "bybit_BTCUSDT_15m_fold4",
        "data_dir": Path("storage/kline"),
        "target_columns": ["log_return", "volatility", "max_drawdown"],
        #"target_columns": ["log_return"],
        #"target_columns": ["direction"],
        "exclude_columns": [],
        "run": run_catboost,
        "do_hyperpram_search": False,
        "two_stage": False,
        "is_classification": False,  # 回帰問題の場合はFalse
    },
}



def train_model(model_name):
    config = MODEL_CONFIGS[model_name]
    targets = config["target_columns"]
    target_str = "_".join(targets) if len(targets) > 1 else targets[0]
    print(f"Starting training for model: {model_name} (target: {target_str})")
    
    # 対応する run_folds 関数を実行
    config["run"](config)
    
    print(f"Training completed for model: {model_name}.")

def main():
    model_names = list(MODEL_CONFIGS.keys())
    # 各モデルを並列実行
    with ProcessPoolExecutor(max_workers=len(model_names)) as executor:
        futures = [executor.submit(train_model, model_name) for model_name in model_names]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Exception during training: {e}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time

    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60

    print(f"\nTotal execution time: {hours}時間 {minutes}分 {seconds:.2f}秒")
