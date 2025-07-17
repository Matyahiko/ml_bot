from __future__ import annotations
from typing import List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DropHighCorrFeatures(BaseEstimator, TransformerMixin):
    """
    byチャッピー
    #TODO:チェック
    相関係数が閾値を超える特徴量を列ごとに削除する Transformer

    Parameters
    ----------
    threshold : float, default=0.95
        |corr| がこの値より大きいペアのうち後ろ側の列を除去
    """

    def __init__(self, threshold: float = 0.95) -> None:
        self.threshold = threshold
        self.to_drop_: List[str] = []

    # ---------- fit ----------
    def fit(self, X: pd.DataFrame, y=None):  # y は未使用
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 絶対値相関行列
        corr = X.corr(numeric_only=True).abs()

        # 上三角行列だけを対象に重複計算を回避
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        # 閾値を超えた列を記録（後ろ側の列を落とす）
        self.to_drop_ = [
            col for col in upper.columns if (upper[col] > self.threshold).any()
        ]
        return self

    # ---------- transform ----------
    def transform(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        return X.drop(columns=self.to_drop_, errors="ignore").copy()

    # ---------- get_feature_names_out ----------
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [c for c in input_features if c not in self.to_drop_]
