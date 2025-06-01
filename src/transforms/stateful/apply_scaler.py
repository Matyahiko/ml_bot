from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class ApplyScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_name='standard', **scaler_kwargs):
        self.scaler_name = scaler_name
        self.scaler_kwargs = scaler_kwargs

    def _init_scaler(self):
        name = self.scaler_name.lower()
        if name == 'standard':
            return StandardScaler(**self.scaler_kwargs)
        elif name == 'minmax':
            return MinMaxScaler(**self.scaler_kwargs)
        elif name == 'robust':
            return RobustScaler(**self.scaler_kwargs)
        elif name == 'normalizer':
            return Normalizer(**self.scaler_kwargs)
        else:
            raise ValueError(f'未知のスケーラー: {self.scaler_name}')

    def fit(self, X, y=None):
        self._scalers_ = {}
        syms = X.columns.get_level_values(0).unique()
        for sym in syms:
            scaler = self._init_scaler()
            scaler.fit(X[sym].values)
            self._scalers_[sym] = scaler
        return self

    def transform(self, X, y=None):
        out = {}
        for sym, scaler in self._scalers_.items():
            data = scaler.transform(X[sym].values)
            for i, sub in enumerate(X[sym].columns):
                out[(sym, sub)] = pd.Series(data[:, i], index=X.index)
        feat_df = pd.DataFrame(out, index=X.index)
        feat_df.columns = pd.MultiIndex.from_tuples(feat_df.columns)
        return feat_df[X.columns]  # 列順を保持
