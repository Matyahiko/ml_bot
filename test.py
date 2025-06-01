import pandas as pd

df = pd.read_pickle("ml/storage/kline/bybit_BTCUSDT_15m_fold0_train.pkl")

print(df.info())
print(df.shape)

df.to_csv("ml/storage/kline/bybit_BTCUSDT_15m_fold0_train.csv")
#df = df[["log_return","volatility"]]