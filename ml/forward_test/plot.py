import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

def plot_results(cerebro, strat, config, logger):
    # 初期資金と最終評価額、総リターンの計算
    initial_cash = config['cash']
    final_portfolio_value = cerebro.broker.getvalue()
    total_return_pct = ((final_portfolio_value - initial_cash) / initial_cash) * 100

    logger.info(f"最終評価額: {final_portfolio_value:.2f}")
    logger.info(f"総リターン: {total_return_pct:.2f}%")

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

    # パフォーマンスレポート（日本語）
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

    # --- 詳細なTrade Analyzerの結果を再帰的にプリント ---
    def recursive_print(d, prefix=""):
        for key, value in d.items():
            if isinstance(value, dict):
                recursive_print(value, prefix + key + ".")
            else:
                print(f"{prefix}{key} : {value}")

    print("=== 詳細なTrade Analyzerの結果 ===")
    recursive_print(trade_analyzer)

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
    # cerebro.datas[0] から終値データを取得する例
    # ※バックトレーダーではヒストリカルデータのインデックスに負の値を用います
    data = cerebro.datas[0]
    dates = [pd.to_datetime(data.datetime.date(i)) for i in range(-len(data), 0)]
    close_prices = [data.close[i] for i in range(-len(data), 0)]
    close_series = pd.Series(close_prices, index=dates)
    close_series.sort_index(inplace=True)

    # --- サブプロットで図と表を作成 ---
    # 左側にグラフ、右側に表（グリッド比率 3:1）
    fig, (ax_plot, ax_table) = plt.subplots(1, 2, figsize=(20, 12), gridspec_kw={'width_ratios': [3, 1]})

    # 左側：日次損益（棒グラフ）、累積利益（赤の折れ線グラフ）、終値（緑の折れ線グラフ）を重ねて表示
    ax_plot.bar(daily_profit_series.index, daily_profit_series, color='skyblue', label='日次損益')
    ax_plot.plot(cumulative_profit_series.index, cumulative_profit_series, color='red', marker='o', linewidth=2, label='累積利益')
    ax_plot.plot(close_series.index, close_series, color='green', marker='x', linewidth=2, label='終値')
    ax_plot.set_title('日次損益、累積利益、終値の推移', fontsize=16)
    ax_plot.set_xlabel('日付', fontsize=14)
    ax_plot.set_ylabel('金額', fontsize=14)
    ax_plot.legend(fontsize=12)
    for tick in ax_plot.get_xticklabels():
        tick.set_rotation(45)
    ax_plot.grid(True)

    # 右側：パフォーマンスレポートの表（日本語）
    ax_table.axis('off')
    ax_table.set_title("パフォーマンス概要", fontsize=16, pad=20)

    table_data = []
    for key, value in report.items():
        if isinstance(value, float):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        table_data.append([key, value_str])

    table = ax_table.table(cellText=table_data,
                           colLabels=["項目", "値"],
                           cellLoc='left',
                           loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)  # 行の高さを拡大

    plt.tight_layout()

    # 図の保存
    profit_loss_file = 'forward_test/profit_loss.png'
    fig.savefig(profit_loss_file)
    logger.info(f"累積損益グラフと表を含む画像を {profit_loss_file} に保存しました。")

    print(f'最終評価額: {final_portfolio_value:.2f}')

    # Cerebro のチャートプロット（必要に応じて）
    plot_list = cerebro.plot()
    # cerebro.plot() はリストのリストで返されるため、最初の図を取り出す
    fig_chart = plot_list[0][0]
    fig_chart.savefig('forward_test/forwardtest.png')
