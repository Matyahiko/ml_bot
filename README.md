# ML Bot

機械学習を用いた暗号資産取引戦略の研究・検証プロジェクト

## プロジェクト概要

このプロジェクトは、暗号資産市場データに対して機械学習モデルを適用し、取引戦略の有効性を検証するための研究環境です。Hydraによる柔軟な設定管理とOptunaによるハイパーパラメータ最適化を組み合わせ、効率的な実験を可能にします。

## ディレクトリ構造

```
ml_bot/
├── research/               # 研究・実験用コード
│   ├── data/              # データ格納ディレクトリ
│   └── src/
│       ├── conf/          # Hydra設定ファイル
│       │   ├── model/     # モデル設定
│       │   ├── pipeline/  # パイプライン設定
│       │   ├── simulator/ # シミュレーター設定
│       │   └── strategy/  # 戦略設定
│       ├── labels/        # ラベリング戦略
│       ├── train/         # 学習・評価モジュール
│       ├── transforms/    # 特徴量エンジニアリング
│       │   ├── stateless/ # 非状態依存の変換
│       │   └── stateful/  # 状態依存の変換
│       ├── utils/         # ユーティリティ関数
│       ├── validation/    # クロスバリデーション
│       └── 1_data_pipeline.py  # メインパイプライン
├── docker-compose.yml
├── dockerfile
├── pyproject.toml
└── requirements.txt
```

## 主要機能

### データパイプライン (`1_data_pipeline.py`)

データ読み込み、特徴量エンジニアリング、ラベリング、モデル学習を統合したパイプライン

**主な処理:**
- 時系列データの分割（MovingWindowKFold）
- 特徴量の追加（時間特徴、テクニカル指標、ラグ特徴、ローリング統計量）
- スケーリング（Robust Scaler等）
- バックテストベースのラベリング（Vectorbt）
- LightGBMモデルによる学習・評価
- Optunaによるハイパーパラメータ探索

### 特徴量変換

**非状態依存（Stateless）:**
- 時間特徴量（周期的エンコーディング）
- テクニカル指標（RSI, MACD, Bollinger Bands等）
- ラグ特徴量
- ローリング統計量（移動平均、標準偏差等）

**状態依存（Stateful）:**
- スケーリング（Robust, Standard, MinMax等）

### ラベリング

- Vectorbtを用いたバックテストシミュレーション
- 戦略ベースのシグナル生成（Squeeze Momentum等）
- リターン・ボラティリティ・最大ドローダウンの計算

## 使い方

### 基本的な実行

```bash
cd research/src
python 1_data_pipeline.py
```

### Hydra設定のオーバーライド

```bash
# データパスを変更
python 1_data_pipeline.py pipeline.data_path=/path/to/data

# モデル設定を変更
python 1_data_pipeline.py model=lgbm_huber
```

### Optunaによるハイパーパラメータ探索

```bash
python 1_data_pipeline.py -m \
  pipeline.search.rolling_windows=5,10,20 \
  pipeline.search.lag=1,2,3
```

## 設定ファイル

設定は `research/src/conf/` 配下のYAMLファイルで管理されています：

- `config.yaml`: ルート設定
- `pipeline/base.yaml`: パイプラインの基本設定
- `pipeline/search.yml`: 探索対象パラメータ
- `model/`: モデル固有の設定
- `strategy/`: 取引戦略の設定
- `simulator/`: シミュレーター設定

## 依存関係

主要なライブラリ：
- Python 3.10-3.12
- pandas, scikit-learn
- LightGBM
- Hydra（設定管理）
- Optuna（ハイパーパラメータ最適化）
- Vectorbt（バックテスト）
- ta-lib（テクニカル指標）

詳細は `requirements.txt` または `pyproject.toml` を参照してください。

## 開発ガイドライン

プロジェクトの開発方針は `agent.md` に記載されています。

## ライセンス

（ライセンス情報を追加する場合はここに記載）
