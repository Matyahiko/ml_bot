# modules/prepocess/normalization.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler

def fit_transform_scaler(train_df: pd.DataFrame, 
                         method: str = 'standard', 
                         winsor_limits: tuple = (0.05, 0.05)):
    """
    学習データに対して前処理を行い、fitしたScaler（またはScaler情報）を返します。
    対象となるのは、timestamp 以外の数値カラム（float, int）全てです。
    
    Parameters
    ----------
    train_df : pd.DataFrame
        学習データ
    method : str, optional
        前処理の方法:
         - 'standard'   : StandardScaler をそのまま適用（デフォルト）
         - 'winsorize'  : 各数値カラムを winsorizing（下位 winsor_limits[0]、上位 winsor_limits[1]）後に StandardScaler を適用
         - 'robust'     : RobustScaler を適用
    winsor_limits : tuple, optional
        winsorizing の閾値（下側, 上側）のパーセンタイル（例: (0.05, 0.05) なら 5% 以下・5% 以上をそれぞれ切り上げ/切り下げ）
        ※ method が 'winsorize' の場合にのみ使用されます。
    
    Returns
    -------
    train_df : pd.DataFrame
        前処理後の学習データ（数値カラムが変換済み）
    scaler_obj : object
        後続の transform に用いる scaler オブジェクト。method が 'winsorize' の場合は dict で
        { "scaler": StandardScaler, "method": "winsorize", "winsor_thresholds": {col: (lower, upper), ...} } となります。
        それ以外は、StandardScaler または RobustScaler のインスタンスを返します。
    """
    # 数値カラムのみを選択（必要に応じて 'timestamp' など除外してください）
    numeric_cols = train_df.select_dtypes(include=["float", "int"]).columns

    if method == 'standard':
        scaler = StandardScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        return train_df, scaler

    elif method == 'winsorize':
        # 各カラムごとに winsorizing 用の閾値を算出し、クリップ処理を実施
        winsor_thresholds = {}
        for col in numeric_cols:
            lower = train_df[col].quantile(winsor_limits[0])
            upper = train_df[col].quantile(1 - winsor_limits[1])
            winsor_thresholds[col] = (lower, upper)
            train_df[col] = train_df[col].clip(lower=lower, upper=upper)
        # その後 StandardScaler を適用
        scaler = StandardScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        # transform 時に winsorizing の閾値も利用できるよう dict にまとめる
        scaler_obj = {
            "scaler": scaler,
            "method": "winsorize",
            "winsor_thresholds": winsor_thresholds,
            "winsor_limits": winsor_limits
        }
        return train_df, scaler_obj

    elif method == 'robust':
        scaler = RobustScaler()
        train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
        return train_df, scaler

    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'standard', 'winsorize', or 'robust'.")


def transform_scaler(test_df: pd.DataFrame, scaler_obj, winsor_limits: tuple = (0.05, 0.05)):
    """
    学習データで fit した scaler_obj を用いて、テスト（または検証）データを変換します。
    """
    numeric_cols = test_df.select_dtypes(include=["float", "int"]).columns

    # winsorize の場合は、学習時に算出した閾値を用いてテストデータにも同様のクリップ処理を実施
    if isinstance(scaler_obj, dict) and scaler_obj.get("method") == "winsorize":
        winsor_thresholds = scaler_obj.get("winsor_thresholds", {})
        for col in numeric_cols:
            if col in winsor_thresholds:
                lower, upper = winsor_thresholds[col]
                # .loc を使って値の代入することで SettingWithCopyWarning を回避
                test_df.loc[:, col] = test_df.loc[:, col].clip(lower=lower, upper=upper)
        # その後、StandardScaler の transform を適用
        test_df.loc[:, numeric_cols] = scaler_obj["scaler"].transform(test_df.loc[:, numeric_cols])
        return test_df

    else:
        # 'standard' あるいは 'robust' の場合は、直接 transform を適用
        test_df.loc[:, numeric_cols] = scaler_obj.transform(test_df.loc[:, numeric_cols])
        return test_df