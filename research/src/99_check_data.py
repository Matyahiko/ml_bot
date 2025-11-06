import pandas as pd
import matplotlib.pyplot as plt
import os

# ファイルパス
train_path = '/app/ml_bot/tmp/fold4/train.parquet'
test_path = '/app/ml_bot/tmp/fold4/test.parquet'
output_dir = '/app/ml_bot/tmp/fold4'

# データの読み込み
train_df = pd.read_parquet(train_path)
test_df = pd.read_parquet(test_path)

# 統計量の取得
train_stats = train_df['ret_2'].describe()
test_stats = test_df['ret_2'].describe()

# signal ≠ 0 の割合を追加
train_signal_ratio = (train_df['signal'] != 0).mean()
test_signal_ratio = (test_df['signal'] != 0).mean()
train_stats['signal_nonzero_ratio'] = train_signal_ratio
test_stats['signal_nonzero_ratio'] = test_signal_ratio

# 統計量を保存
train_stats.to_csv(os.path.join(output_dir, 'ret_2_train_stats.csv'))
test_stats.to_csv(os.path.join(output_dir, 'ret_2_test_stats.csv'))

# 外れ値を除外：1%〜99%の範囲に限定
lower_bound = min(train_df['ret_2'].quantile(0.01), test_df['ret_2'].quantile(0.01))
upper_bound = max(train_df['ret_2'].quantile(0.99), test_df['ret_2'].quantile(0.99))

train_trimmed = train_df['ret_2'][(train_df['ret_2'] >= lower_bound) & (train_df['ret_2'] <= upper_bound)]
test_trimmed = test_df['ret_2'][(test_df['ret_2'] >= lower_bound) & (test_df['ret_2'] <= upper_bound)]

# ヒストグラムの描画と保存
plt.figure(figsize=(10, 5))
plt.hist(train_trimmed, bins=100, alpha=0.5, density=True, label='Train')
plt.hist(test_trimmed, bins=100, alpha=0.5, density=True, label='Test')
plt.title('Trimmed Histogram of ret_2 (1% - 99%)')
plt.xlabel('ret_2')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ret_2_histogram_trimmed.png'))
plt.close()
