import talib
import pandas as pd

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    symbols = df.columns.get_level_values(0).unique()
    symbol_frames = []

    for symbol in symbols:
        # 元のOHLCVデータ
        data = df[symbol]
        open_ = data['Open']
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']

        # 価格の中心（高値と安値の中間）
        hilo = (high + low) / 2
        ind = {}

        # ボリンジャーバンド
        ub, mb, lb = talib.BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        ind['BBANDS_upperband'] = (ub - hilo) / close
        ind['BBANDS_middleband'] = (mb - hilo) / close
        ind['BBANDS_lowerband'] = (lb - hilo) / close

        # 各種移動平均
        ind['DEMA'] = (talib.DEMA(close, timeperiod=30) - hilo) / close
        ind['EMA'] = (talib.EMA(close, timeperiod=30) - hilo) / close
        ind['HT_TRENDLINE'] = (talib.HT_TRENDLINE(close) - hilo) / close
        ind['KAMA'] = (talib.KAMA(close, timeperiod=30) - hilo) / close
        ind['MA'] = (talib.MA(close, timeperiod=30, matype=0) - hilo) / close
        ind['MIDPOINT'] = (talib.MIDPOINT(close, timeperiod=14) - hilo) / close
        ind['SMA'] = (talib.SMA(close, timeperiod=30) - hilo) / close
        ind['T3'] = (talib.T3(close, timeperiod=5, vfactor=0) - hilo) / close
        ind['TEMA'] = (talib.TEMA(close, timeperiod=30) - hilo) / close
        ind['TRIMA'] = (talib.TRIMA(close, timeperiod=30) - hilo) / close
        ind['WMA'] = (talib.WMA(close, timeperiod=30) - hilo) / close

        # 線形回帰
        ind['LINEARREG'] = (talib.LINEARREG(close, timeperiod=14) - close) / close
        ind['LINEARREG_INTERCEPT'] = (talib.LINEARREG_INTERCEPT(close, timeperiod=14) - close) / close

        # 出来高・価格変換
        ind['AD'] = talib.AD(high, low, close, volume) / close
        ind['ADOSC'] = talib.ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10) / close

        # オシレーター
        ind['APO'] = talib.APO(close, fastperiod=12, slowperiod=26, matype=0)
        ht_ip, ht_qd = talib.HT_PHASOR(close)
        ind['HT_PHASOR_inphase'] = ht_ip
        ind['HT_PHASOR_quadrature'] = ht_qd

        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        ind['MACD_macd'] = macd
        ind['MACD_macdsignal'] = macdsignal
        ind['MACD_macdhist'] = macdhist

        ind['MOM'] = talib.MOM(close, timeperiod=10)
        ind['MFI'] = talib.MFI(high, low, close, volume, timeperiod=14)
        ind['CMO'] = talib.CMO(close, timeperiod=14)
        ind['PPO'] = talib.PPO(close, fastperiod=12, slowperiod=26, matype=0)
        ind['ROC'] = talib.ROC(close, timeperiod=10)
        ind['ROCP'] = talib.ROCP(close, timeperiod=10)
        ind['ROCR'] = talib.ROCR(close, timeperiod=10)
        ind['ROCR100'] = talib.ROCR100(close, timeperiod=10)

        # ストキャスティクス
        sk, sd = talib.STOCH(high, low, close,
                             fastk_period=5, slowk_period=3, slowk_matype=0,
                             slowd_period=3, slowd_matype=0)
        ind['STOCH_slowk'] = sk
        ind['STOCH_slowd'] = sd
        fk, fd = talib.STOCHF(high, low, close,
                              fastk_period=5, fastd_period=3, fastd_matype=0)
        ind['STOCHF_fastk'] = fk
        ind['STOCHF_fastd'] = fd
        srsik, srsid = talib.STOCHRSI(close,
                                      timeperiod=14, fastk_period=5,
                                      fastd_period=3, fastd_matype=0)
        ind['STOCHRSI_fastk'] = srsik
        ind['STOCHRSI_fastd'] = srsid

        ind['LINEARREG_SLOPE'] = talib.LINEARREG_SLOPE(close, timeperiod=14) / close
        ind['MINUS_DM'] = talib.MINUS_DM(high, low, timeperiod=14) / close
        ind['OBV'] = talib.OBV(close, volume) / close
        ind['PLUS_DM'] = talib.PLUS_DM(high, low, timeperiod=14) / close
        ind['STDDEV'] = talib.STDDEV(close, timeperiod=5, nbdev=1) / close
        ind['TRANGE'] = talib.TRANGE(high, low, close) / close

        # トレンド・モメンタム指標
        ind['ADX'] = talib.ADX(high, low, close, timeperiod=14)
        ind['ADXR'] = talib.ADXR(high, low, close, timeperiod=14)
        adown, aup = talib.AROON(high, low, timeperiod=14)
        ind['AROON_aroondown'] = adown
        ind['AROON_aroonup'] = aup
        ind['AROONOSC'] = talib.AROONOSC(high, low, timeperiod=14)
        ind['BOP'] = talib.BOP(open_, high, low, close)
        ind['CCI'] = talib.CCI(high, low, close, timeperiod=14)
        ind['DX'] = talib.DX(high, low, close, timeperiod=14)
        ind['RSI'] = talib.RSI(close, timeperiod=14)
        ind['TRIX'] = talib.TRIX(close, timeperiod=30)
        ind['ULTOSC'] = talib.ULTOSC(high, low, close,
                                     timeperiod1=7, timeperiod2=14, timeperiod3=28)
        ind['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)

        # ボラティリティ
        ind['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        ind['NATR'] = talib.NATR(high, low, close, timeperiod=14)

        # ヒルベルト変換
        ind['HT_DCPERIOD'] = talib.HT_DCPERIOD(close)
        ind['HT_DCPHASE'] = talib.HT_DCPHASE(close)
        hsine, hlead = talib.HT_SINE(close)
        ind['HT_SINE_sine'] = hsine
        ind['HT_SINE_leadsine'] = hlead
        ind['HT_TRENDMODE'] = talib.HT_TRENDMODE(close)

        # 統計指標
        ind['BETA'] = talib.BETA(high, low, timeperiod=5)
        ind['CORREL'] = talib.CORREL(high, low, timeperiod=30)
        ind['LINEARREG_ANGLE'] = talib.LINEARREG_ANGLE(close, timeperiod=14)

        # MESA指標
        mama, fama = talib.MAMA(close, fastlimit=0.5, slowlimit=0.05)
        ind['MAMA'] = (mama - hilo) / close
        ind['FAMA'] = (fama - hilo) / close

        # 価格指標
        ind['MIDPRICE'] = (talib.MIDPRICE(high, low, timeperiod=14) - hilo) / close
        ind['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2) / close
        ind['SAREXT'] = talib.SAREXT(
            high, low,
            startvalue=0, offsetonreverse=0,
            accelerationinitlong=0.02, accelerationlong=0.02, accelerationmaxlong=0.2,
            accelerationinitshort=0.02, accelerationshort=0.02, accelerationmaxshort=0.2
        ) / close
        ind['MEDPRICE'] = talib.MEDPRICE(high, low) / close
        ind['TYPPRICE'] = talib.TYPPRICE(high, low, close) / close
        ind['WCLPRICE'] = talib.WCLPRICE(high, low, close) / close
        
        
        # VWAP（5期間の例）
        vwap = (close * volume).rolling(window=5).sum() / volume.rolling(window=5).sum()
        ind['VWAP'] = (vwap - hilo) / close  # スケール合わせ

        # Chaikin Money Flow（20期間）
        clv = ((close - low) - (high - close)) / (high - low + 1e-9)   # 0除算対策
        cmf = (clv * volume).rolling(window=20).sum() / \
            (volume.rolling(window=20).sum() + 1e-9)
        ind['CMF'] = cmf

        # 指標データフレームを作成し、マルチインデックスの列を回復
        ind_df = pd.DataFrame(ind, index=df.index)
        ind_df.columns = pd.MultiIndex.from_product([[symbol], ind_df.columns])

        # 元データと結合
        combined = pd.concat([df[symbol], ind_df[symbol]], axis=1)
        combined.columns = pd.MultiIndex.from_product([[symbol], combined.columns])

        symbol_frames.append(combined)

    # 全てのシンボルを横に連結し、NaNを削除、マルチインデックスのまま返す
    if symbol_frames:
        result = pd.concat(symbol_frames, axis=1)
        result.dropna(inplace=True)
        return result

    return df.copy()
