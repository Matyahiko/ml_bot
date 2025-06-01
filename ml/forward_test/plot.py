from rich import print
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

def plot_results(cerebro, strat, config, logger):
    
    #srebroをプロット
    figs = cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False)
    # cerebro.plot returns a nested list of figures; here we pick the first one
    fig = figs[0][0]
    fig.savefig('forward_test/visualization.png')
    
    # 初期資金と最終評価額、総リターンの計算
    initial_cash = config['cash']
    final_portfolio_value = cerebro.broker.getvalue()
    total_return_pct = ((final_portfolio_value - initial_cash) / initial_cash) * 100

    print(f"最終評価額: {final_portfolio_value:.2f}")
    print(f"総リターン: {total_return_pct:.2f}%")

    # 各種アナライザーの取得
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    trade_analyzer = strat.analyzers.trade_analyzer.get_analysis()
    annual_return = strat.analyzers.annual_return.get_analysis()
    sqn = strat.analyzers.sqn.get_analysis()

    # TradeAnalyzer による取引数の計算
    if 'total' in trade_analyzer and 'closed' in trade_analyzer['total']:
        total_closed = trade_analyzer['total']['closed']
        won_total = trade_analyzer.get('won', {}).get('total', 0)
        lost_total = trade_analyzer.get('lost', {}).get('total', 0)
        win_rate = (won_total / total_closed * 100) if total_closed > 0 else 'N/A'
    else:
        total_closed = 0
        won_total = 0
        lost_total = 0
        win_rate = 'N/A'

    # パフォーマンスレポート（日本語）を辞書にまとめ、printで表示
    report = {
        '最終評価額': final_portfolio_value,
        '総リターン (%)': total_return_pct,
        'シャープレシオ': sharpe.get('sharperatio', 'N/A') if sharpe else 'N/A',
        '最大ドローダウン (%)': drawdown.get('max', {}).get('drawdown', 'N/A') if drawdown else 'N/A',
        '総取引回数': total_closed,
        '勝ちトレード数': won_total,
        '負けトレード数': lost_total,
        '勝率 (%)': win_rate,
        '年次リターン (%)': annual_return.get('yearly', 'N/A') if annual_return else 'N/A',
        'SQN': sqn.get('sqn', 'N/A') if sqn else 'N/A',
    }

    print("=== パフォーマンスレポート ===")
    for key, value in report.items():
        print(f"{key}: {value}")

    # --- 累積損益のプロット作成 ---
    # アナライザーから得られる時系列リターンを利用
    time_return = strat.analyzers.time_return.get_analysis()
    time_return_series = pd.Series(time_return)
    time_return_series.index = pd.to_datetime(time_return_series.index)
    time_return_series.sort_index(inplace=True)
    print("時系列リターン（一部）:")
    print(time_return_series.head())

    # 資産の計算：初期資金 * (1 + 時系列リターンの累積積)
    asset_value_series = initial_cash * (1 + time_return_series).cumprod()
    # 累積利益の計算：資産 - 初期資金
    cumulative_profit_series = asset_value_series - initial_cash
    # 日次損益の計算（前日の資産との差分）
    daily_profit_series = asset_value_series.diff().fillna(0)

    # --- 終値の取得 ---
    data = cerebro.datas[0]
    dates = [pd.to_datetime(data.datetime.date(i)) for i in range(-len(data), 0)]
    close_prices = [data.close[i] for i in range(-len(data), 0)]
    close_series = pd.Series(close_prices, index=dates)
    close_series.sort_index(inplace=True)

    # --- 単一のプロットで図を作成 ---
    fig, ax = plt.subplots(figsize=(20, 12))

    # 日次損益（棒グラフ）、累積利益（赤の折れ線グラフ）、終値（緑の折れ線グラフ）を重ねて表示
    ax.bar(daily_profit_series.index, daily_profit_series, color='skyblue', label='日次損益')
    ax.plot(cumulative_profit_series.index, cumulative_profit_series, color='red', marker='o', linewidth=2, label='累積利益')
    ax.plot(close_series.index, close_series, color='green', marker='x', linewidth=2, label='終値')
    ax.set_title('日次損益、累積利益、終値の推移', fontsize=16)
    ax.set_xlabel('日付', fontsize=14)
    ax.set_ylabel('金額', fontsize=14)
    ax.legend(fontsize=12)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
    ax.grid(True)

    plt.tight_layout()

    # 図の保存（forward_test/forwardtest.png は作成しない）
    profit_loss_file = 'forward_test/profit_loss.png'
    fig.savefig(profit_loss_file)
    print(f"累積損益グラフを {profit_loss_file} に保存しました。")

    print(f'最終評価額: {final_portfolio_value:.2f}')

    # --- 各取引ごとの結果をプロット ---
    trade_dates = []
    trade_pnl = []
    if 'trades' in trade_analyzer:
        for trade in trade_analyzer['trades'].values():
            if 'pnl' in trade and 'dt' in trade:
                trade_dates.append(pd.to_datetime(trade['dt']))
                trade_pnl.append(trade['pnl']['net']['total'])

    trade_results_df = pd.DataFrame({'損益': trade_pnl}, index=trade_dates)
    trade_results_df.sort_index(inplace=True)
    cumulative_trade_pnl = trade_results_df['損益'].cumsum()

    plt.figure(figsize=(12, 6))
    plt.bar(trade_results_df.index, trade_results_df['損益'], color='skyblue', label='各取引の損益')
    plt.plot(cumulative_trade_pnl.index, cumulative_trade_pnl, color='red', marker='o', linewidth=2, label='累積損益')
    plt.title('実際の取引に基づく累積損益の推移', fontsize=16)
    plt.xlabel('日付', fontsize=14)
    plt.ylabel('損益', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('forward_test/actual_trade_profit_loss.png')
