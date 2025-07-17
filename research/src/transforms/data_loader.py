from pathlib import Path
import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path: str | Path, exchange: str, interval: str, file_format: str, cache: bool = False):
        self.data_path = Path(data_path)
        self.interval = interval
        self.exchange = exchange
        self.file_format = file_format.lower()
        self.cache = cache
        # シンボルごとのキャッシュ用辞書
        self._cache: dict[str, pd.DataFrame] = {}
    
    def _file_path(self, symbol: str) -> Path:
        return self.data_path / f"{self.exchange}_{symbol}_{self.interval}.{self.file_format}"

    def _load_single(self, symbol: str) -> pd.DataFrame:
        # キャッシュ確認
        if self.cache and symbol in self._cache:
            return self._cache[symbol]

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
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        df = df.set_index('timestamp')
        # 列名をフラット化: symbol_field の形式
        df.columns = [f"{symbol}_{col}" for col in df.columns]
        df.index.name = 'timestamp'

        if self.cache:
            self._cache[symbol] = df
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
