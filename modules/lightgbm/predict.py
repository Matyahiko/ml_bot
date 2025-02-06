import pandas as pd
import lightgbm as lgb
import numpy as np

def predict(df: pd.DataFrame, model_path) -> pd.DataFrame:
    model = lgb.Booster(model_file=model_path)
    predictions = model.predict(df)
    # クラス（0, 1, 2）へのマッピング
    mapped_classes = np.argmax(predictions, axis=1)
    print(set(mapped_classes))
    
    print(mapped_classes)
    df['predictions'] = mapped_classes
    return df