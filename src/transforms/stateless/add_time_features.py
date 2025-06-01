import numpy as np
import polars as pl


def add_time_features_cyc(df: pl.DataFrame) -> pl.DataFrame:
    """
    時間特徴量と周期変換
    """
    ts = pl.col("timestamp")
    WEEK_PERIOD = 52  # 訓練では 52 週周期とみなす（53 週の年は誤差と割り切る）

    return df.with_columns(
        [
            #hour (0–23) 
            (ts.dt.hour() * 2 * np.pi / 24).sin().alias("hour_sin"),
            (ts.dt.hour() * 2 * np.pi / 24).cos().alias("hour_cos"),

            #day-of-week (Monday=0)
            ((ts.dt.weekday() - 1) * 2 * np.pi / 7).sin().alias("dow_sin"),
            ((ts.dt.weekday() - 1) * 2 * np.pi / 7).cos().alias("dow_cos"),

            #month (0–11)
            ((ts.dt.month() - 1) * 2 * np.pi / 12).sin().alias("month_sin"),
            ((ts.dt.month() - 1) * 2 * np.pi / 12).cos().alias("month_cos"),

            #ISO week number (0–52)
            ((ts.dt.week() - 1) * 2 * np.pi / WEEK_PERIOD).sin().alias("week_sin"),
            ((ts.dt.week() - 1) * 2 * np.pi / WEEK_PERIOD).cos().alias("week_cos"),

            #quarter (0–3)
            ((ts.dt.quarter() - 1) * 2 * np.pi / 4).sin().alias("quarter_sin"),
            ((ts.dt.quarter() - 1) * 2 * np.pi / 4).cos().alias("quarter_cos"),
        ]
    )
