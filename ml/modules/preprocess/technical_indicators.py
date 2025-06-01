# technical_indicators.py
import talib
import pandas as pd

def technical_indicators(df, prefix):
    open_, high, low, close, volume = (
        df[f"{prefix}_Open"],
        df[f"{prefix}_High"],
        df[f"{prefix}_Low"],
        df[f"{prefix}_Close"],
        df[f"{prefix}_Volume"],
    )
    hilo = (high + low) / 2

    ind = {}

    ind[f"{prefix}_BBANDS_upperband"], ind[f"{prefix}_BBANDS_middleband"], ind[f"{prefix}_BBANDS_lowerband"] = talib.BBANDS(close, 5, 2, 2, 0)
    ind[f"{prefix}_BBANDS_upperband"]  = (ind[f"{prefix}_BBANDS_upperband"]  - hilo) / close
    ind[f"{prefix}_BBANDS_middleband"] = (ind[f"{prefix}_BBANDS_middleband"] - hilo) / close
    ind[f"{prefix}_BBANDS_lowerband"]  = (ind[f"{prefix}_BBANDS_lowerband"]  - hilo) / close

    ind[f"{prefix}_DEMA"]  = (talib.DEMA(close, 30)  - hilo) / close
    ind[f"{prefix}_EMA"]   = (talib.EMA(close, 30)   - hilo) / close
    ind[f"{prefix}_HT_TRENDLINE"] = (talib.HT_TRENDLINE(close) - hilo) / close
    ind[f"{prefix}_KAMA"]  = (talib.KAMA(close, 30)  - hilo) / close
    ind[f"{prefix}_MA"]    = (talib.MA(close, 30, 0) - hilo) / close
    ind[f"{prefix}_MIDPOINT"] = (talib.MIDPOINT(close, 14) - hilo) / close
    ind[f"{prefix}_SMA"]   = (talib.SMA(close, 30)   - hilo) / close
    ind[f"{prefix}_T3"]    = (talib.T3(close, 5, 0)  - hilo) / close
    ind[f"{prefix}_TEMA"]  = (talib.TEMA(close, 30)  - hilo) / close
    ind[f"{prefix}_TRIMA"] = (talib.TRIMA(close, 30) - hilo) / close
    ind[f"{prefix}_WMA"]   = (talib.WMA(close, 30)   - hilo) / close
    ind[f"{prefix}_LINEARREG"]            = (talib.LINEARREG(close, 14)            - close) / close
    ind[f"{prefix}_LINEARREG_INTERCEPT"]  = (talib.LINEARREG_INTERCEPT(close, 14)  - close) / close

    # ----------------------------------------------------------------------
    # Volume & price transforms (スケール維持)
    # ----------------------------------------------------------------------
    ind[f"{prefix}_AD"]    = talib.AD(high, low, close, volume) / close
    ind[f"{prefix}_ADOSC"] = talib.ADOSC(high, low, close, volume, 3, 10) / close

    # ----------------------------------------------------------------------
    # Oscillators (正規化を削除)
    # ----------------------------------------------------------------------
    ind[f"{prefix}_APO"] = talib.APO(close, 12, 26, 0)

    ht_ip, ht_qd = talib.HT_PHASOR(close)
    ind[f"{prefix}_HT_PHASOR_inphase"]    = ht_ip
    ind[f"{prefix}_HT_PHASOR_quadrature"] = ht_qd

    macd, macds, macdh = talib.MACD(close, 12, 26, 9)
    ind[f"{prefix}_MACD_macd"]        = macd
    ind[f"{prefix}_MACD_macdsignal"]  = macds
    ind[f"{prefix}_MACD_macdhist"]    = macdh

    ind[f"{prefix}_MOM"] = talib.MOM(close, 10)

    ind[f"{prefix}_MFI"]      = talib.MFI(high, low, close, volume, 14)
    ind[f"{prefix}_CMO"]      = talib.CMO(close, 14)
    ind[f"{prefix}_PPO"]      = talib.PPO(close, 12, 26, 0)
    ind[f"{prefix}_ROC"]      = talib.ROC(close, 10)
    ind[f"{prefix}_ROCP"]     = talib.ROCP(close, 10)
    ind[f"{prefix}_ROCR"]     = talib.ROCR(close, 10)
    ind[f"{prefix}_ROCR100"]  = talib.ROCR100(close, 10)

 
    sk, sd = talib.STOCH(high, low, close, 5, 3, 0, 3, 0)
    ind[f"{prefix}_STOCH_slowk"], ind[f"{prefix}_STOCH_slowd"] = sk, sd
    fk, fd = talib.STOCHF(high, low, close, 5, 3, 0)
    ind[f"{prefix}_STOCHF_fastk"], ind[f"{prefix}_STOCHF_fastd"] = fk, fd
    srsik, srsid = talib.STOCHRSI(close, 14, 5, 3, 0)
    ind[f"{prefix}_STOCHRSI_fastk"], ind[f"{prefix}_STOCHRSI_fastd"] = srsik, srsid

    ind[f"{prefix}_LINEARREG_SLOPE"] = talib.LINEARREG_SLOPE(close, 14) / close
    ind[f"{prefix}_MINUS_DM"] = talib.MINUS_DM(high, low, 14) / close
    ind[f"{prefix}_OBV"]      = talib.OBV(close, volume) / close
    ind[f"{prefix}_PLUS_DM"]  = talib.PLUS_DM(high, low, 14) / close
    ind[f"{prefix}_STDDEV"]   = talib.STDDEV(close, 5, 1) / close
    ind[f"{prefix}_TRANGE"]   = talib.TRANGE(high, low, close) / close

    ind[f"{prefix}_ADX"]  = talib.ADX(high, low, close, 14)
    ind[f"{prefix}_ADXR"] = talib.ADXR(high, low, close, 14)
    adown, aup = talib.AROON(high, low, 14)
    ind[f"{prefix}_AROON_aroondown"], ind[f"{prefix}_AROON_aroonup"] = adown, aup
    ind[f"{prefix}_AROONOSC"] = talib.AROONOSC(high, low, 14)
    ind[f"{prefix}_BOP"]      = talib.BOP(open_, high, low, close)
    ind[f"{prefix}_CCI"]      = talib.CCI(high, low, close, 14)
    ind[f"{prefix}_DX"]       = talib.DX(high, low, close, 14)
    ind[f"{prefix}_RSI"]      = talib.RSI(close, 14)
    ind[f"{prefix}_TRIX"]     = talib.TRIX(close, 30)
    ind[f"{prefix}_ULTOSC"]   = talib.ULTOSC(high, low, close, 7, 14, 28)
    ind[f"{prefix}_WILLR"]    = talib.WILLR(high, low, close, 14)

    ind[f"{prefix}_ATR"]  = talib.ATR(high, low, close, 14)
    ind[f"{prefix}_NATR"] = talib.NATR(high, low, close, 14)

    ind[f"{prefix}_HT_DCPERIOD"] = talib.HT_DCPERIOD(close)
    ind[f"{prefix}_HT_DCPHASE"]  = talib.HT_DCPHASE(close)
    hsine, hlead = talib.HT_SINE(close)
    ind[f"{prefix}_HT_SINE_sine"], ind[f"{prefix}_HT_SINE_leadsine"] = hsine, hlead
    ind[f"{prefix}_HT_TRENDMODE"] = talib.HT_TRENDMODE(close)

    ind[f"{prefix}_BETA"]                = talib.BETA(high, low, 5)
    ind[f"{prefix}_CORREL"]              = talib.CORREL(high, low, 30)
    ind[f"{prefix}_LINEARREG_ANGLE"]     = talib.LINEARREG_ANGLE(close, 14)

    mama, fama = talib.MAMA(close, 0.5, 0.05)
    ind[f"{prefix}_MAMA"] = (mama - hilo) / close
    ind[f"{prefix}_FAMA"] = (fama - hilo) / close
    ind[f"{prefix}_MIDPRICE"] = (talib.MIDPRICE(high, low, 14) - hilo) / close
    ind[f"{prefix}_SAR"]      = talib.SAR(high, low, 0.02, 0.2) / close
    ind[f"{prefix}_SAREXT"]   = talib.SAREXT(
        high, low,
        startvalue=0, offsetonreverse=0,
        accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
        accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2
    ) / close

    ind[f"{prefix}_MEDPRICE"] = talib.MEDPRICE(high, low) / close
    ind[f"{prefix}_TYPPRICE"] = talib.TYPPRICE(high, low, close) / close
    ind[f"{prefix}_WCLPRICE"] = talib.WCLPRICE(high, low, close) / close

    ind_df = pd.DataFrame(ind, index=df.index)
    df = pd.concat([df, ind_df], axis=1).copy()
    df.dropna(inplace=True)

    # 例として RSI を保存 (必要なら変更)
    df[[f"{prefix}_RSI"]].to_csv(f"storage/kline/temp/{prefix}_technical_indicators.csv")
    return df