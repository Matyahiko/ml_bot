# Kline Data Pipeline 仕様書

## 概要
このデータパイプラインは、複数の暗号資産の時系列価格データ（Klineデータ）を処理し、機械学習モデルのトレーニングとテスト用のデータセットを生成するためのものです。パイプラインは、データの読み込み、特徴量エンジニアリング、教師ラベルの生成、時系列クロスバリデーションによるデータ分割を行います。

## 入力データ要件
- **データ形式**: CSV形式
- **ファイル命名規則**: `bybit_{シンボル名}_15m.csv`
- **必須カラム**: `timestamp`（UTC時間）および価格データ（OHLCV等）
- **保存場所**: `raw_data/`ディレクトリ

## 処理ステップ

### 1. データの読み込みと統合
- 指定された複数の暗号資産シンボル（例：BTCUSDT, ETHUSDT, XRPUSDT）のデータを読み込む
- 各シンボルのデータにプレフィックスを付けて区別（例：`BTCUSDT_open`, `ETHUSDT_close`）
- タイムスタンプを基準に内部結合（inner join）してデータを統合
- 出力ディレクトリの初期化（`storage/kline/`および`storage/kline/temp/`）

### 2. 前処理と特徴量エンジニアリング
以下の特徴量を各シンボルに対して生成します：

#### 時間ベースの特徴量
- `add_time_features`関数を使用して、時間に関連する特徴量を追加
  - 曜日、時間帯、月などの時間的特徴

#### テクニカル指標
- `technical_indicators`関数を使用して、各シンボルに対してテクニカル指標を計算
  - RSI、MACD、ボリンジャーバンドなどの一般的なテクニカル指標

#### 統計的特徴量
- `add_rolling_statistics`関数を使用して、ローリングウィンドウベースの統計量を計算
  - ウィンドウサイズ: 5, 10, 20
  - 移動平均、標準偏差、最大値、最小値などの統計量

#### ラグ特徴量
- `add_lag_features`関数を使用して、過去の値を特徴量として追加
  - ラグ値: 1, 2, 3（過去1〜3期間の値）

#### 教師ラベル（目的変数）の生成
- `backtest_labeling_run`関数を使用して、バックテストベースのラベルを生成
  - ログリターン
  - ボラティリティ
  - 最大ドローダウン

以下の教師ラベル生成関数はコメントアウトされていますが、必要に応じて有効化できます：
- `add_price_direction_labelç: n期間後の価格変動方向（上昇/下降）
- `add_price_change_rate`: n期間後の価格変化率
- `add_future_volatility`: n期間後のボラティリティ（n≧2が必要）
- `add_excess_return`: n期間後の超過リターン（ベンチマークに対する相対リターン）

####　特徴量の削減
- `PCAでpca_feature_reduction`:全数値型カラムにPCAを適用し、99%の累積説明分散比率を保持

#### データクリーニング
- 欠損値の除去
- データのシャッフル（時系列の順序は保持）

### 3. 時系列クロスバリデーションによるデータ分割
- `MovingWindowKFold`クラスを使用して、時系列を考慮したクロスバリデーション分割を実行
- デフォルトで5分割（num_folds=5）
- 各分割（フォールド）ごとにトレーニングセットとテストセットを生成
- 並列処理によるフォールド処理の高速化（joblib.Parallel使用）

### 4. 処理済みデータの保存
- 各フォールドのトレーニングセットとテストセットをpickle形式で保存
- 保存先: `storage/kline/{filename}_fold{fold}_{subset_name}.pkl`
  - filename: 基本ファイル名（デフォルト: `bybit_BTCUSDT_15m`）
  - fold: フォールド番号（0〜num_folds-1）
  - subset_name: `train`または`test`

## 出力データ
- **形式**: pickle (.pkl)
- **保存場所**: `storage/kline/`ディレクトリ
- **ファイル命名規則**: `{filename}_fold{fold}_{subset_name}.pkl`
- **内容**: 特徴量と教師ラベルを含む前処理済みのデータフレーム

## 設定オプション
`DataPipeline`クラスの初期化パラメータ：

- **symbols**: 処理する暗号資産シンボルのリスト（例：`['BTCUSDT', 'ETHUSDT', 'XRPUSDT']`）
- **base_path**: 入力データの基本パス（デフォルト: `'raw_data'`）
- **filename**: 出力ファイルの基本名（デフォルト: `'bybit_BTCUSDT_15m'`）
- **num_folds**: クロスバリデーションのフォールド数（デフォルト: `5`）
- **n_jobs**: 並列処理に使用するCPUコア数（デフォルト: `-1`（全コア使用））

## キャッシュ機能
- joblib.Memoryを使用したキャッシュ機能
- キャッシュ保存先: `./joblib_cache/`
- 不要な場合は削除可能

## 実行時間計測
- パイプライン全体の実行時間を計測し、時間、分、秒単位で表示

## 使用例
```python
symbols = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT']
pipeline = DataPipeline(
    symbols=symbols,
    base_path='raw_data',
    filename='bybit_BTCUSDT_15m',
    num_folds=5,
    n_jobs=-1
)
pipeline.run()
```
