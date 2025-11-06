import sys
from typing import Any, Iterator
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

#https://blog.amedama.jp/entry/time-series-cv
#https://zenn.dev/skata/articles/20241204-tscv
class MovingWindowKFold(TimeSeriesSplit):

    def __init__(self, clipping: bool = False, *args: Any, **kwargs: Any) -> None:

        super().__init__(*args, **kwargs)
        self.clipping: bool = clipping

    def split(self, X: pd.DataFrame, *args: Any, **kwargs: Any) -> Iterator[tuple[list[int], list[int]]]:

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        # 各フォールドの最小長を追跡する
        train_min: int = sys.maxsize
        test_min: int = sys.maxsize

        # 全レベルのインデックスでソート
        sorted_df: pd.DataFrame = X.sort_index()

        # ソート後の行ラベル -> 元DF上の iloc 行番号 へのマッピング
        orig_to_sorted_pos = X.index.get_indexer(sorted_df.index)

        # sklearn の分割を、ソート済み DataFrame に対して実行
        for train_idx, test_idx in super().split(sorted_df, *args, **kwargs):
            train_pos = orig_to_sorted_pos[train_idx]
            test_pos: np.ndarray = orig_to_sorted_pos[test_idx]

            if self.clipping:
                train_min = min(train_min, len(train_pos))
                test_min = min(test_min, len(test_pos))
                train_pos = train_pos[-train_min:]
                test_pos = test_pos[-test_min:]

            yield train_pos.tolist(), test_pos.tolist()
