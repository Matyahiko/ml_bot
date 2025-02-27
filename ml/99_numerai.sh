#!/usr/bin/env python3

import os
import numerapi
from pathlib import Path

# 保存先ディレクトリを作成
save_dir = Path("storage/numerai")
save_dir.mkdir(parents=True, exist_ok=True)

# NumeraiのAPIクライアントを初期化
print("Numerai API クライアントを初期化しています...")
napi = numerapi.NumerAPI(verbosity="info")

# 利用可能なデータセットのリストを取得して表示
print("利用可能なデータセットを確認中...")
datasets = napi.list_datasets()
print("\n".join(datasets))

# ダウンロードするデータセット
files_to_download = [
    "v5.0/train.parquet",
    "v5.0/validation.parquet",
    "v5.0/live.parquet",
    "v5.0/features.json",
    "v5.0/train_benchmark_models.parquet",
    "v5.0/live_example_preds.parquet"
]

# 各データセットをダウンロード
for dataset in files_to_download:
    filename = os.path.basename(dataset)
    save_path = save_dir / filename
    
    print(f"{filename} をダウンロード中...")
    try:
        napi.download_dataset(dataset, str(save_path))
        print(f"{filename} を {save_path} に保存しました")
    except Exception as e:
        print(f"{filename} のダウンロード中にエラーが発生しました: {e}")

print("すべてのダウンロードが完了しました")