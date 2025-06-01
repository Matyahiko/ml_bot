import pandas as pd
import os
from datetime import datetime
import pathlib
import shutil
from rich import print

class StorageUtils:
    def __init__(self, top_dir: str = "/app/ml_bot/ml/artifacts"):
        self.top_dir = pathlib.Path(top_dir)
        self.top_dir.mkdir(parents=True, exist_ok=True)

    def reset_dir(self):
        if self.top_dir.exists():
            shutil.rmtree(self.top_dir)
        self.top_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(self, data, file_name: str):
        path = self._build_path(file_name, ".csv")
        data.to_csv(path, index=False)

    def save_parquet(self, data, file_name: str):
        path = self._build_path(file_name, ".parquet")
        data.to_parquet(path, index=False)

    def save_pickle(self, data, file_name: str):
        path = self._build_path(file_name, ".pkl")
        data.to_pickle(path)

    def _build_path(self, file_name, default_ext):
        path = self.top_dir / file_name
        if path.suffix != default_ext:
            path = path.with_suffix(default_ext)
        return path
