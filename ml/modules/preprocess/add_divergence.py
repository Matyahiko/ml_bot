import itertools
import pandas as pd

def add_divergence(df: pd.DataFrame, sym: str) -> pd.DataFrame:

    # 指定シンボルのテクニカル指標カラムを抽出
    tech_cols = [col for col in df.columns if col.startswith(f"{sym}_")]
    
    # 各ペアの差分を辞書に格納
    divergence_data = {}
    for col1, col2 in itertools.combinations(tech_cols, 2):
        divergence_col = f"div_{col1}_{col2}"
        divergence_data[divergence_col] = df[col1] - df[col2]
    
    # 辞書から新たなDataFrameを生成（全ての差分カラムを一括で取得）
    divergence_df = pd.DataFrame(divergence_data, index=df.index)
    
    # 元のDataFrameと新規差分DataFrameをpd.concatで横方向に結合
    new_df = pd.concat([df, divergence_df], axis=1)
    
    # 新たなDataFrameの断片化を防ぐためにcopyを実施
    new_df = new_df.copy()
    
    return new_df
