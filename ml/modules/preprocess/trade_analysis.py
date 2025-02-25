import numpy as np
import pandas as pd
from rich import print

def analyze_trades(trade_data):
    """
    各期間（トレード期間および非トレード期間）ごとに、対数リターン、ボラティリティ、最大ドローダウンを算出し、
    その結果を元データの各行に割り当てる。  
    ※トレードが発生していない期間は、log_return, volatility, max_drawdown をすべて 0 とする。
    
    Parameters:
        trade_data (pd.DataFrame): インデックスが datetime で、'open', 'high', 'low', 'close',
                                   'trade_flag', 'trade_type' などを含む DataFrame。
    
    Returns:
        pd.DataFrame: 元の行数と同じインデックスを持ち、各行に log_return, volatility, max_drawdown の値が割り当てられた DataFrame。
    """
    # 連続する trade_flag の変化でグループ化
    group_id = (trade_data['trade_flag'] != trade_data['trade_flag'].shift()).cumsum()
    
    # 元データと同じインデックスの結果用 DataFrame を作成
    result_df = pd.DataFrame(index=trade_data.index)
    result_df['log_return'] = np.nan
    result_df['volatility'] = np.nan
    result_df['max_drawdown'] = np.nan
    
    # グループごとに計算
    for _, group in trade_data.groupby(group_id):
        start_time = group.index[0]
        end_time = group.index[-1]
        flag = group.iloc[0]['trade_flag']
        
        if flag:  # トレード期間の場合
            entry_price = group.iloc[0]['open']
            trade_type = group.iloc[0]['trade_type']
            exit_price = group.iloc[-1]['close']
            
            # 対数リターンの計算
            if trade_type == 'long':
                log_return = np.log(exit_price / entry_price)
            elif trade_type == 'short':
                log_return = np.log(entry_price / exit_price)
            else:
                log_return = np.nan
            
            # 各バー間の対数リターンからボラティリティを計算
            intra_returns = []
            prev_price = group.iloc[0]['open']
            for close in group['close']:
                r = np.log(close / prev_price)
                intra_returns.append(r)
                prev_price = close
            volatility = np.std(intra_returns) if intra_returns else np.nan
            
            # 最大ドローダウンの計算
            price_series = group['close'].values
            if trade_type == 'long':
                running_max = np.maximum.accumulate(price_series)
                drawdowns = (running_max - price_series) / running_max
            elif trade_type == 'short':
                running_min = np.minimum.accumulate(price_series)
                drawdowns = (price_series - running_min) / running_min
            else:
                drawdowns = np.array([np.nan])
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else np.nan
            
            # 計算結果をグループ内の全行に適用
            result_df.loc[group.index, 'log_return'] = log_return
            result_df.loc[group.index, 'volatility'] = volatility
            result_df.loc[group.index, 'max_drawdown'] = max_drawdown
        else:
            # 非トレード期間の場合は全て 0 を割り当て
            result_df.loc[group.index, 'log_return'] = 0
            result_df.loc[group.index, 'volatility'] = 0
            result_df.loc[group.index, 'max_drawdown'] = 0
            
    # 万が一欠損があれば補完
    result_df['log_return'] = result_df['log_return'].fillna(0)
    result_df['volatility'] = result_df['volatility'].fillna(0)
    result_df['max_drawdown'] = result_df['max_drawdown'].fillna(0)
    
    return result_df
