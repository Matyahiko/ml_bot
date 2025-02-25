# calc_technical_indicators.py
import talib
import pandas as pd

def technical_indicators(df, prefix):
    # テクニカル指標を計算するためのデータを取得
    open_, high, low, close, volume = (
        df[f"{prefix}_Open"],
        df[f"{prefix}_High"],
        df[f"{prefix}_Low"],
        df[f"{prefix}_Close"],
        df[f"{prefix}_Volume"],
    )
    hilo = (high + low) / 2

    # 新しいテクニカル指標を格納する辞書
    indicators = {}

    # 価格(hilo または close)を引いた後、価格(close)で割ることで標準化
    indicators[f'{prefix}_BBANDS_upperband'], indicators[f'{prefix}_BBANDS_middleband'], indicators[f'{prefix}_BBANDS_lowerband'] = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    indicators[f'{prefix}_BBANDS_upperband'] = (indicators[f'{prefix}_BBANDS_upperband'] - hilo) / close
    indicators[f'{prefix}_BBANDS_middleband'] = (indicators[f'{prefix}_BBANDS_middleband'] - hilo) / close
    indicators[f'{prefix}_BBANDS_lowerband'] = (indicators[f'{prefix}_BBANDS_lowerband'] - hilo) / close

    indicators[f'{prefix}_DEMA'] = (talib.DEMA(close, timeperiod=30) - hilo) / close
    indicators[f'{prefix}_EMA'] = (talib.EMA(close, timeperiod=30) - hilo) / close
    indicators[f'{prefix}_HT_TRENDLINE'] = (talib.HT_TRENDLINE(close) - hilo) / close
    indicators[f'{prefix}_KAMA'] = (talib.KAMA(close, timeperiod=30) - hilo) / close
    indicators[f'{prefix}_MA'] = (talib.MA(close, timeperiod=30, matype=0) - hilo) / close
    indicators[f'{prefix}_MIDPOINT'] = (talib.MIDPOINT(close, timeperiod=14) - hilo) / close
    indicators[f'{prefix}_SMA'] = (talib.SMA(close, timeperiod=30) - hilo) / close
    indicators[f'{prefix}_T3'] = (talib.T3(close, timeperiod=5, vfactor=0) - hilo) / close
    indicators[f'{prefix}_TEMA'] = (talib.TEMA(close, timeperiod=30) - hilo) / close
    indicators[f'{prefix}_TRIMA'] = (talib.TRIMA(close, timeperiod=30) - hilo) / close
    indicators[f'{prefix}_WMA'] = (talib.WMA(close, timeperiod=30) - hilo) / close
    indicators[f'{prefix}_LINEARREG'] = (talib.LINEARREG(close, timeperiod=14) - close) / close
    indicators[f'{prefix}_LINEARREG_INTERCEPT'] = (talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close) / close

    # 価格(close)で割ることで標準化
    indicators[f'{prefix}_AD'] = talib.AD(high, low, close, volume) / close
    indicators[f'{prefix}_ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10) / close
    indicators[f'{prefix}_APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0) / close
    ht_phasor_inphase, ht_phasor_quadrature = talib.HT_PHASOR(close)
    indicators[f'{prefix}_HT_PHASOR_inphase'] = ht_phasor_inphase / close
    indicators[f'{prefix}_HT_PHASOR_quadrature'] = ht_phasor_quadrature / close
    indicators[f'{prefix}_LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14) / close
    macd_macd, macd_macdsignal, macd_macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    indicators[f'{prefix}_MACD_macd'] = macd_macd / close
    indicators[f'{prefix}_MACD_macdsignal'] = macd_macdsignal / close
    indicators[f'{prefix}_MACD_macdhist'] = macd_macdhist / close
    indicators[f'{prefix}_MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14) / close
    indicators[f'{prefix}_MOM'] = talib.MOM(close, timeperiod=10) / close
    indicators[f'{prefix}_OBV'] = talib.OBV(close, volume) / close
    indicators[f'{prefix}_PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14) / close
    indicators[f'{prefix}_STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1) / close
    indicators[f'{prefix}_TRANGE'] = talib.TRANGE(high, low, close) / close

    indicators[f'{prefix}_ADX'] = talib.ADX(high, low, close, timeperiod=14)
    indicators[f'{prefix}_ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
    aroon_aroondown, aroon_aroonup = talib.AROON(high, low, timeperiod=14)
    indicators[f'{prefix}_AROON_aroondown'] = aroon_aroondown
    indicators[f'{prefix}_AROON_aroonup'] = aroon_aroonup
    indicators[f'{prefix}_AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
    indicators[f'{prefix}_BOP'] = talib.BOP(open_, high, low, close)
    indicators[f'{prefix}_CCI'] = talib.CCI(high, low, close, timeperiod=14)
    indicators[f'{prefix}_DX'] = talib.DX(high, low, close, timeperiod=14)
    indicators[f'{prefix}_MFI'] = talib.MFI(high, low, close, volume, timeperiod=14) / close
    indicators[f'{prefix}_MINUS_DI'] = talib.MINUS_DI(high, low, close, timeperiod=14) / close
    indicators[f'{prefix}_PLUS_DI'] = talib.PLUS_DI(high, low, close, timeperiod=14) / close
    indicators[f'{prefix}_RSI'] = talib.RSI(close, timeperiod=14)
    stoch_slowk, stoch_slowd = talib.STOCH(
        high, low, close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
    )
    indicators[f'{prefix}_STOCH_slowk'] = stoch_slowk
    indicators[f'{prefix}_STOCH_slowd'] = stoch_slowd
    stochf_fastk, stochf_fastd = talib.STOCHF(
        high, low, close, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    indicators[f'{prefix}_STOCHF_fastk'] = stochf_fastk
    indicators[f'{prefix}_STOCHF_fastd'] = stochf_fastd
    stochrsi_fastk, stochrsi_fastd = talib.STOCHRSI(
        close, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0
    )
    indicators[f'{prefix}_STOCHRSI_fastk'] = stochrsi_fastk
    indicators[f'{prefix}_STOCHRSI_fastd'] = stochrsi_fastd
    indicators[f'{prefix}_TRIX'] = talib.TRIX(close, timeperiod=30)
    indicators[f'{prefix}_ULTOSC'] = talib.ULTOSC(high, low, close, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    indicators[f'{prefix}_WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

    indicators[f'{prefix}_ATR'] = talib.ATR(high, low, close, timeperiod=14)
    indicators[f'{prefix}_NATR'] = talib.NATR(high, low, close, timeperiod=14)

    indicators[f'{prefix}_HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
    indicators[f'{prefix}_HT_DCPHASE'] = talib.HT_DCPHASE(close)
    ht_sine_sine, ht_sine_leadsine = talib.HT_SINE(close)
    indicators[f'{prefix}_HT_SINE_sine'] = ht_sine_sine / close
    indicators[f'{prefix}_HT_SINE_leadsine'] = ht_sine_leadsine / close
    indicators[f'{prefix}_HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

    indicators[f'{prefix}_BETA'] = talib.BETA(high, low, timeperiod=5)
    indicators[f'{prefix}_CORREL'] = talib.CORREL(high, low, timeperiod=30)

    indicators[f'{prefix}_LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)

    # Overlap Studiesの追加
    # MAMA (MESA Adaptive Moving Average)
    mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
    indicators[f'{prefix}_MAMA'] = (mama - hilo) / close
    indicators[f'{prefix}_FAMA'] = (fama - hilo) / close  # Optional: FAMA (Following MAMA)

    # MIDPRICE (Midpoint Price over period)
    indicators[f'{prefix}_MIDPRICE'] = (talib.MIDPRICE(high, low, timeperiod=14) - hilo) / close

    # SAR (Parabolic SAR)
    indicators[f'{prefix}_SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2) / close

    # SAREXT (Parabolic SAR - Extended)
    indicators[f'{prefix}_SAREXT'] = talib.SAREXT(
        high, low,
        startvalue=0, offsetonreverse=0,
        accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
        accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2
    ) / close

    # Momentum Indicatorsの追加
    # CMO (Chande Momentum Oscillator)
    indicators[f'{prefix}_CMO'] = talib.CMO(close, timeperiod=14) / close

    # PPO (Percentage Price Oscillator)
    indicators[f'{prefix}_PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0) / close

    # ROC (Rate of Change)
    indicators[f'{prefix}_ROC'] = talib.ROC(close, timeperiod=10) / close

    # ROCP (Rate of Change Percentage)
    indicators[f'{prefix}_ROCP'] = talib.ROCP(close, timeperiod=10) / close

    # ROCR (Rate of Change Ratio)
    indicators[f'{prefix}_ROCR'] = talib.ROCR(close, timeperiod=10) / close

    # ROCR100 (Rate of Change Ratio 100 Scale)
    indicators[f'{prefix}_ROCR100'] = talib.ROCR100(close, timeperiod=10) / close

    # Price Transformの追加
    # AVGPRICE (Average Price)
    # 注: コメントアウトされているため、必要に応じて追加してください

    # MEDPRICE (Median Price)
    indicators[f'{prefix}_MEDPRICE'] = talib.MEDPRICE(high, low) / close

    # TYPPRICE (Typical Price)
    indicators[f'{prefix}_TYPPRICE'] = talib.TYPPRICE(high, low, close) / close

    # WCLPRICE (Weighted Close Price)
    indicators[f'{prefix}_WCLPRICE'] = talib.WCLPRICE(high, low, close) / close

    # 一時的なDataFrameとして新しい指標を作成
    indicators_df = pd.DataFrame(indicators, index=df.index)

    # 元のDataFrameに新しい指標を結合
    df = pd.concat([df, indicators_df], axis=1)

    # フラグメンテーションを解消
    df = df.copy()

    df.dropna(inplace=True)
    
    # 保存時にもプレフィックスを考慮する場合は、必要に応じてファイル名を変更してください
    df.to_csv(f'storage/kline/{prefix}_technical_indicators.csv')
    
    return df
