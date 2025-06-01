from pathlib import Path
import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path: str | Path, exchenge: str, interval: str, file_format: str, cache: bool= False):
        self.data_path = Path(data_path)
        self.interval = interval
        self.exchenge = exchenge
        self.file_format = file_format.lower()
        self.cache = cache
    
    def _file_path(self, symbol: str) -> Path:
        return self.data_path / f"{self.exchenge}_{symbol}_{self.interval}.{self.file_format}"

    def _load_single(self, symbol: str) -> pd.DataFrame:
        path = self._file_path(symbol)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        loaders = {
            "parquet": pd.read_parquet,
            "csv": pd.read_csv,
            "pickle": pd.read_pickle,
        }
        loader = loaders.get(self.file_format)
        if loader is None:
            raise ValueError(f"Unsupported file format: {self.file_format}")

        df = loader(path)
        # タイムスタンプを datetime index に
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize(None)
        df = df.set_index('timestamp')
        # 列名を MultiIndex に変更
        df.columns = pd.MultiIndex.from_product(
            [[symbol], df.columns],
            names=['symbol', 'field']
        )
        df.index.name = 'timestamp'
        return df
        
    def load(self, symbols: str | list[str]) -> pd.DataFrame:

        # 単一シンボルならリスト化
        if isinstance(symbols, str):
            symbols = [symbols]

        # 各シンボルごとに DataFrame を取得
        dfs = [self._load_single(sym) for sym in symbols]

        # タイムスタンプを揃えて結合
        combined = pd.concat(dfs, axis=1).sort_index(axis=1)
        return combined

