import matplotlib.pyplot as plt  # 可視化のためのライブラリをインポート
import japanize_matplotlib

def split_time_series(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Returns:
    - train (pd.Series or pd.DataFrame): 訓練セット
    - val (pd.Series or pd.DataFrame): 検証セット
    - test (pd.Series or pd.DataFrame): テストセット
    """
    total = len(dataset)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train = dataset.iloc[:train_end]
    val = dataset.iloc[train_end:val_end]
    test = dataset.iloc[val_end:]
    
    return train, val, test

def visualize_split(dataset, train, val, test):

    plt.figure(figsize=(15, 6))
    
    # 元のデータをプロット
    plt.plot(dataset.index, dataset.values, label='全データ', color='gray')
    
    # 訓練セットの範囲を塗りつぶす
    plt.axvspan(train.index.min(), train.index.max(), color='blue', alpha=0.3, label='訓練セット')
    
    # 検証セットの範囲を塗りつぶす
    plt.axvspan(val.index.min(), val.index.max(), color='green', alpha=0.3, label='検証セット')
    
    # テストセットの範囲を塗りつぶす
    plt.axvspan(test.index.min(), test.index.max(), color='red', alpha=0.3, label='テストセット')
    
    plt.xlabel('時間')
    plt.ylabel('値')
    plt.title('時系列データの分割')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Save/fig/split.png')
