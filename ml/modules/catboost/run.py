import os
import sys

import pandas as pd

from modules.catboost.catboost_classifier_and_regressor import CatBoostClassifierAndMultiRegressor
from modules.catboost.plot_metrics import plot_metrics


def load_data(filename: str, data_dir: str, target_columns: list, num_folds: int = None, fold: int = None):
    if num_folds is not None and fold is not None:
        train = pd.read_pickle(os.path.join(data_dir, f"{filename}_{fold}_train.pkl"))
        test = pd.read_pickle(os.path.join(data_dir, f"{filename}_{fold}_test.pkl"))
    else:
        train = pd.read_pickle(os.path.join(data_dir, f"{filename}_train.pkl"))
        test = pd.read_pickle(os.path.join(data_dir, f"{filename}_test.pkl"))

    # 元のDataFrameをそのまま x_train, x_test として返す
    x_train = train.copy()
    x_test = test.copy()

    # ターゲットデータは別途抽出しておく
    y_train = train[target_columns].copy()
    y_test = test[target_columns].copy()

    return x_train, y_train, x_test, y_test

def run(config: dict):
    num_folds = config.get("num_folds")
    filename = config["filename"]
    data_dir = config["data_dir"]
    target_columns = config["target_columns"]
    do_hyperparam_search = config.get("do_hyperparam_search", False)
    
    
    if do_hyperparam_search:
        pass
    else:
        x_train, y_train, x_test, y_test = load_data(filename, data_dir, target_columns)
        model = CatBoostClassifierAndMultiRegressor(target_columns=target_columns)
        model.train(x_train, y_train, x_test, y_test)
        metrics = model.evaluate(x_test, y_test)
        model.save_models()
        plot_metrics(metrics, save_dir="plots/catboost")
        
        
