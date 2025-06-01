import numpy as np
import polars as pl

def add_time_features(df: pl.DataFrame) -> pl.DataFrame:
    ts = pl.col("timestamp")
    
    df = df.with_columns({
        ts.dt.year().alias("year"),
        ts.dt.month().alias("month"),
        ts.dt.day().alias("day"),
        ts.dt.hour().alias("hour"),
        ts.dt.minute().alias("minute"),
        ts.dt.weekday().alias("day_of_week"),

    })
    
    df = df.with_columns({
        
    })