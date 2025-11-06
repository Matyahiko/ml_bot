from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Type

@dataclass
class SqueezeMomentumConfig:
    bb_period: int = 6           # ボリンジャー/KC の期間
    kc_multiplier: float = 0.9   # KC 用 ATR 乗数
    bb_multiplier: float = 1.2   # BB 用 σ 乗数
    linreg_period: int = 8       # モメンタム傾き計算の窓長
    stop_loss: float = 0.025     # 2.5 %
    take_profit: float = 0.03    # 3.0 %
    time_exit: int = 3           # n 本後クローズ
    threshold: float = 0.012     # |momentum| > threshold
    