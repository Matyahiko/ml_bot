import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from fracdiff import fdiff
from joblib import Parallel, delayed
import multiprocessing

def process_column_initial_adf(column, series, significance_level):
    try:
        result = adfuller(series.dropna())
        is_non_stationary = result[1] > significance_level
        return (column, {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        }, is_non_stationary)
    except Exception as e:
        print(f"初期ADF検定に失敗しました - {column}: {e}")
        return (column, {
            'ADF Statistic': np.nan,
            'p-value': np.nan,
            'Critical Values': {}
        }, False)

def process_column_final_adf(column, series, significance_level):
    try:
        result = adfuller(series.dropna())
        return (column, {
            'ADF Statistic': result[0],
            'p-value': result[1],
            'Critical Values': result[4]
        })
    except Exception as e:
        print(f"最終ADF検定に失敗しました - {column}: {e}")
        return (column, {
            'ADF Statistic': np.nan,
            'p-value': np.nan,
            'Critical Values': {}
        })

def process_fractional_diff(column, series, d):
    try:
        return (column, fdiff(series.values, n=d))
    except Exception as e:
        print(f"分数階差分に失敗しました - {column}: {e}")
        return (column, pd.Series([np.nan]*len(series)))

def fractional_difference_and_adf(df, d=0.35, significance_level=0.05, n_jobs=-1):
    """
    データフレーム内の非定数列に対してADF検定を行い、非定常な列に分数階差分を適用します。
    
    Parameters:
        df (pd.DataFrame): 入力データフレーム。
        d (float): 分数階差分の次数。デフォルトは0.35。
        significance_level (float): ADF検定の有意水準。デフォルトは0.05。
        n_jobs (int): 並列実行するジョブの数。-1は全てのコアを使用。
    
    Returns:
        pd.DataFrame: 分数階差分を適用したデータフレーム。
    """
    print("分数階差分とADF検定を実行します...")
    # 定数列を自動的に除外
    non_constant_columns = [col for col in df.columns if df[col].nunique() > 1]
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
    
    if constant_columns:
        print(f"定数列のため処理をスキップします: {constant_columns}")
    
    # 初期ADF検定を並列実行
    initial_results = {}
    non_stationary_columns = []
    
    num_cores = multiprocessing.cpu_count()
    print(f"使用するCPUコア数: {num_cores}")
    
    initial_adf_results = Parallel(n_jobs=n_jobs)(
        delayed(process_column_initial_adf)(col, df[col], significance_level)
        for col in non_constant_columns
    )
    
    for col, res, is_non_stat in initial_adf_results:
        initial_results[col] = res
        if is_non_stat:
            non_stationary_columns.append(col)
    
    # 非定常な列に対して分数階差分を適用（並列実行）
    if non_stationary_columns:
        print(f"非定常な列に対して分数階差分を適用します: {non_stationary_columns}")
        fracdiff_results = Parallel(n_jobs=n_jobs)(
            delayed(process_fractional_diff)(col, df[col], d)
            for col in non_stationary_columns
        )
        
        df_fracdiff = pd.DataFrame(index=df.index)
        for col, diff_series in fracdiff_results:
            df_fracdiff[col] = diff_series
        
        for col in non_stationary_columns:
            df[col] = df_fracdiff[col]
    
    # 最終ADF検定を並列実行
    final_results = {}
    final_adf_results = Parallel(n_jobs=n_jobs)(
        delayed(process_column_final_adf)(col, df[col], significance_level)
        for col in non_constant_columns
    )
    
    for col, res in final_adf_results:
        final_results[col] = res
    
    # 結果の表示
    print("初期ADF検定結果:")
    display_results(initial_results, significance_level)
    
    print("\n分数階差分適用後のADF検定結果:")
    display_results(final_results, significance_level)
    
    return df

def display_results(results, significance_level):
    """
    ADF検定の結果を表示します。
    
    Parameters:
        results (dict): ADF検定の結果。
        significance_level (float): 有意水準。
    """
    df_results = pd.DataFrame(results).T
    df_results['Is Stationary'] = df_results['p-value'] < significance_level
    df_results['Critical Value (5%)'] = df_results['Critical Values'].apply(
        lambda x: x.get('5%', np.nan) if isinstance(x, dict) else np.nan
    )
    df_results = df_results.drop('Critical Values', axis=1)
    print(df_results.to_string())