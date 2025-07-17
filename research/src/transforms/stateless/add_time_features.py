import numpy as np
import pandas as pd

def add_time_features_cyc(df: pd.DataFrame) -> pd.DataFrame:
    """
    時間特徴量と周期変換
    """
 
    ts = df.index
    WEEK_PERIOD = 52  # 訓練では 52 週周期とみなす（53 週の年は誤差と割り切る）

    features = {
        "hour_sin": np.sin(ts.hour * 2 * np.pi / 24),
        "hour_cos": np.cos(ts.hour * 2 * np.pi / 24),
        "dow_sin": np.sin((ts.weekday - 1) * 2 * np.pi / 7),
        "dow_cos": np.cos((ts.weekday - 1) * 2 * np.pi / 7),
        "month_sin": np.sin((ts.month - 1) * 2 * np.pi / 12),
        "month_cos": np.cos((ts.month - 1) * 2 * np.pi / 12),
        "week_sin": np.sin((ts.isocalendar().week - 1) * 2 * np.pi / WEEK_PERIOD),
        "week_cos": np.cos((ts.isocalendar().week - 1) * 2 * np.pi / WEEK_PERIOD),
        "quarter_sin": np.sin((ts.quarter - 1) * 2 * np.pi / 4),
        "quarter_cos": np.cos((ts.quarter - 1) * 2 * np.pi / 4),
    }
    
    for name, series in features.items():
        df[f"time_{name}"] = series

    return df

