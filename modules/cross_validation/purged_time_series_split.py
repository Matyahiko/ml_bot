# modules/cross_validation/purged_time_series_split.py

from sklearn.model_selection import BaseCrossValidator
import numpy as np

class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    時系列データに対するクロスバリデーションで、パージとエンバーゴを適用します。
    
    Parameters
    ----------
    n_splits : int, default=5
        分割数。
    purge_size : int, default=0
        トレーニングセットとテストセットの間に挟むギャップのサイズ（サンプル数）。
    embargo_size : int, default=0
        テストセットの後に無視するデータのサイズ（サンプル数）。
    """
    def __init__(self, n_splits=5, purge_size=0, embargo_size=0):
        self.n_splits = n_splits
        self.purge_size = purge_size
        self.embargo_size = embargo_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        indices = np.arange(n_samples)
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_start = stop + self.purge_size
            test_end = test_start + fold_size
            if test_end > n_samples:
                test_end = n_samples
            train_indices = indices[start:stop]
            test_indices = indices[test_start:test_end]
            yield train_indices, test_indices
            current = stop
