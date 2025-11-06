import talib
import pandas as pd
from typing import Literal, Optional, List

def add_technical_indicators(
    df: pd.DataFrame,
    symbols: list[str],
    *,
    feature_lag: int = 1,
    fillna: Optional[Literal["ffill", "bfill", "both", "zero"]] = "ffill",
    min_non_na: float | None = 0.7
) -> pd.DataFrame:
    """
    フラットなカラム名（BTC_Open, ETH_Close, ...）のDataFrameから
    各シンボルごとにテクニカル指標を計算し、シンボル_指標名で返す。
    """
    out_frames = []

    for symbol in symbols:
        # 元データからシンボルごとのOHLCVを取り出し
        cols = [f"{symbol}_{col}" for col in ["Open", "High", "Low", "Close", "Volume"]]
        if not all(col in df.columns for col in cols):
            # 対象シンボルのデータが無い場合はskip
            continue
        data = df[cols].copy()
        data.columns = ["Open", "High", "Low", "Close", "Volume"]

        open_, high, low, close, volume = (data[c] for c in ['Open','High','Low','Close','Volume'])
        hilo = (high + low) / 2
        ind: dict[str, pd.Series] = {}

        # 例：Bollinger Bands
        ub, mb, lb = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        ind['BBANDS_upperband']  = (ub - hilo) / close
        ind['BBANDS_middleband'] = (mb - hilo) / close
        ind['BBANDS_lowerband']  = (lb - hilo) / close

        # 単一入力の移動平均系など
        for name, func, period in [
            ('DEMA', talib.DEMA, 30), ('EMA', talib.EMA, 30),
            ('KAMA', talib.KAMA, 30), ('MA', talib.MA, 30),
            ('MIDPOINT', talib.MIDPOINT, 14), ('SMA', talib.SMA, 30),
            ('T3', lambda x, timeperiod: talib.T3(x, timeperiod=timeperiod, vfactor=0), 5),
            ('TEMA', talib.TEMA, 30), ('TRIMA', talib.TRIMA, 30),
            ('WMA', talib.WMA, 30)
        ]:
            ind[name] = (func(close, timeperiod=period) - hilo) / close

        ind['HT_TRENDLINE'] = (talib.HT_TRENDLINE(close) - hilo) / close
        ind['LINEARREG'] = (talib.LINEARREG(close, timeperiod=14) - close) / close
        ind['LINEARREG_INTERCEPT'] = (talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close) / close

        ind['AD'] = talib.AD(high, low, close, volume) / close
        ind['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10) / close

        ind['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
        ht_ip, ht_qd = talib.HT_PHASOR(close)
        ind['HT_PHASOR_inphase'], ind['HT_PHASOR_quadrature'] = ht_ip, ht_qd

        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        ind['MACD_macd'], ind['MACD_macdsignal'], ind['MACD_macdhist'] = macd, macdsignal, macdhist

        # MOM, ROC 系
        for name, func, period in [
            ('MOM', talib.MOM, 10),
            ('ROC', talib.ROC, 10),
            ('ROCP', talib.ROCP, 10),
            ('ROCR', talib.ROCR, 10),
            ('ROCR100', talib.ROCR100, 10)
        ]:
            ind[name] = func(close, timeperiod=period)

        # ストキャスティクス
        sk, sd = talib.STOCH(high, low, close, 5, 3, 0, 3, 0)
        ind['STOCH_slowk'], ind['STOCH_slowd'] = sk, sd
        fk, fd = talib.STOCHF(high, low, close, 5, 3, 0)
        ind['STOCHF_fastk'], ind['STOCHF_fastd'] = fk, fd
        srsik, srsid = talib.STOCHRSI(close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        ind['STOCHRSI_fastk'], ind['STOCHRSI_fastd'] = srsik, srsid

        ind['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14) / close
        ind['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14) / close
        ind['OBV'] = talib.OBV(close, volume) / close
        ind['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14) / close
        ind['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1) / close
        ind['TRANGE'] = talib.TRANGE(high, low, close) / close

        ind['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        ind['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        adown, aup = talib.AROON(high, low, timeperiod=14)
        ind['AROON_aroondown'], ind['AROON_aroonup'] = adown, aup
        ind['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
        ind['BOP'] = talib.BOP(open_, high, low, close)
        ind['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        ind['DX'] = talib.DX(high, low, close, timeperiod=14)
        ind['RSI'] = talib.RSI(close, timeperiod=14)
        ind['TRIX'] = talib.TRIX(close, timeperiod=30)
        ind['ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        ind['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        ind['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        ind['NATR'] = talib.NATR(high, low, close, timeperiod=14)

        ind['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
        ind['HT_DCPHASE'] = talib.HT_DCPHASE(close)
        hsine, hlead = talib.HT_SINE(close)
        ind['HT_SINE_sine'], ind['HT_SINE_leadsine'] = hsine, hlead
        ind['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        ind['BETA'] = talib.BETA(high, low, timeperiod=5)
        ind['CORREL'] = talib.CORREL(high, low, timeperiod=30)
        ind['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)

        mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
        ind['MAMA'] = (mama - hilo) / close
        ind['FAMA'] = (fama - hilo) / close

        ind['MIDPRICE'] = (talib.MIDPRICE(high, low, timeperiod=14) - hilo) / close
        ind['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2) / close
        ind['SAREXT'] = talib.SAREXT(
            high, low,
            startvalue=0, offsetonreverse=0,
            accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
            accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2
        ) / close
        ind['MEDPRICE']  = talib.MEDPRICE(high, low) / close
        ind['TYPPRICE']  = talib.TYPPRICE(high, low, close) / close
        ind['WCLPRICE']  = talib.WCLPRICE(high, low, close) / close

        # VWAP & CMF
        vwap = (close * volume).rolling(window=5).sum() / volume.rolling(window=5).sum()
        ind['VWAP'] = (vwap - hilo) / close

        clv = ((close - low) - (high - close)) / (high - low + 1e-9)
        cmf = (clv * volume).rolling(window=20).sum() / (volume.rolling(window=20).sum() + 1e-9)
        ind['CMF'] = cmf

        # MFI
        ind['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)

        # DataFrame化 & シフト
        ind_df = pd.DataFrame(ind, index=df.index)
        if feature_lag:
            ind_df = ind_df.shift(feature_lag)

        # カラム名を symbol_指標名 に
        ind_df.columns = [f"{symbol}_{col}" for col in ind_df.columns]

        # 元データ + 指標を結合
        data.columns = [f"{symbol}_{col}" for col in data.columns]
        sym_out = pd.concat([data, ind_df], axis=1)
        out_frames.append(sym_out)

    if not out_frames:
        return df.copy()

    # 全シンボル結合（カラムはすべて1階層 str）
    result = pd.concat(out_frames, axis=1)

    # 欠損補完
    if fillna:
        if fillna in ("ffill", "both"):
            result.ffill(inplace=True)
        if fillna in ("bfill", "both"):
            result.bfill(inplace=True)
        if fillna == "zero":
            result.fillna(0.0, inplace=True)

    # 行 drop & 割合 print
    before = len(result)
    if min_non_na is not None:
        thresh = int(result.shape[1] * float(min_non_na))
        result.dropna(thresh=thresh, inplace=True)
    after = len(result)
    # pct = 100 * (before - after) / before
    # print(f"[add_technical_indicators] dropped {before - after} rows ({pct:.2f}%)")

    return result
