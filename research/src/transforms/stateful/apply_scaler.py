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
        # timeで始まるカラム以外
        self._target_cols_ = [col for col in X.columns if not str(col).startswith("time")]
        self._other_cols_ = [col for col in X.columns if str(col).startswith("time")]
        self._scaler_ = self._init_scaler()
        self._scaler_.fit(X[self._target_cols_].values)
        return self

    def transform(self, X, y=None):
        X_scaled = X.copy()
        if self._target_cols_:
            scaled_data = self._scaler_.transform(X[self._target_cols_].values)
            X_scaled[self._target_cols_] = scaled_data
        return X_scaled
