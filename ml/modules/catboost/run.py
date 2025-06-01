import os
import sys

import pandas as pd

from modules.catboost.catboost_classifier_and_regressor import CatBoostClassifierAndMultiRegressor
from modules.catboost.catboost_regressor import CatBoostMultiRegressor
from modules.catboost.catboost_classifier import CatBoostMultiClassifier
from modules.catboost.plot_metrics import plot_metrics, export_predictions_to_csv


def load_data(filename: str, data_dir: str, target_columns: list, exclude_columns: list, num_folds: int = None, fold: int = None):
    if num_folds is not None and fold is not None:
        train = pd.read_pickle(os.path.join(data_dir, f"{filename}_{fold}_train.pkl")).drop(columns=exclude_columns)
        test = pd.read_pickle(os.path.join(data_dir, f"{filename}_{fold}_test.pkl")).drop(columns=exclude_columns)
    else:
        train = pd.read_pickle(os.path.join(data_dir, f"{filename}_train.pkl")).drop(columns=exclude_columns)
        test = pd.read_pickle(os.path.join(data_dir, f"{filename}_test.pkl")).drop(columns=exclude_columns)
    
    # ターゲットデータは別途抽出しておく
    y_train = train[target_columns].copy()
    y_test = test[target_columns].copy()
    
    # ターゲット列を削除した特徴量データを作成
    x_train = train.drop(columns=target_columns).copy()
    x_test = test.drop(columns=target_columns).copy()
    
    return x_train, y_train, x_test, y_test

def run(config: dict):
    num_folds = config.get("num_folds")
    filename = config["filename"]
    data_dir = config["data_dir"]
    target_columns = config["target_columns"]
    exclude_columns = config.get("exclude_columns", [])
    do_hyperparam_search = config.get("do_hyperparam_search", False)
    two_stage = config.get("two_stage", False)
    is_classification = config.get("is_classification", False)
    loss_function = "RMSE" if len(target_columns) == 1 else "MultiRMSE"
    
    if is_classification:
        # 分類問題の場合
        if do_hyperparam_search:
            pass
        else:
            x_train, y_train, x_test, y_test = load_data(filename, data_dir, target_columns, exclude_columns)
            model = CatBoostMultiClassifier(target_columns=target_columns)
            model.train(x_train, y_train, x_test, y_test)
            metrics = model.evaluate(x_test, y_test)
            model.save_models()
            model.print_results(detailed=True)
            plot_metrics(metrics, save_dir="plots/catboost")
            export_predictions_to_csv(model, x_test, save_dir="plots/catboost")
    elif two_stage:
        # 二段階モデル（分類+回帰）の場合
        if do_hyperparam_search:
            pass
        else:
            x_train, y_train, x_test, y_test = load_data(filename, data_dir, target_columns, exclude_columns)
            model = CatBoostClassifierAndMultiRegressor(target_columns=target_columns)
            model.train(x_train, y_train, x_test, y_test)
            metrics = model.evaluate(x_test, y_test)
            model.save_models()
            model.print_results(detailed=True)
            plot_metrics(metrics, save_dir="plots/catboost")
            export_predictions_to_csv(model, x_test, save_dir="plots/catboost")
    else:
        # 回帰問題の場合
        if do_hyperparam_search:
            pass
        else:
            x_train, y_train, x_test, y_test = load_data(filename, data_dir, target_columns, exclude_columns)
            model = CatBoostMultiRegressor(target_columns=target_columns, loss_function=loss_function)
            model.train(x_train, y_train, x_test, y_test)
            metrics = model.evaluate(x_test, y_test)
            model.save_models()
            model.print_results(detailed=True)
            plot_metrics(metrics, save_dir="plots/catboost")
            export_predictions_to_csv(model, x_test, save_dir="plots/catboost")
