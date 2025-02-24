import copy
import torch
import logging
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import os
from rich import print

# CatBoost
from catboost import CatBoostRegressor, Pool

# Optuna 関連
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import Trial

# ユーザー定義の前処理済みデータセットなど
from modules.rnn_base.rnn_data_process_multitask import TimeSeriesDataset  # window_sizeなどの時間系列切り出しに利用

# plot
from modules.catboost.plot_multitask import plot_final_training_results_multitask

# R2スコア算出用
from sklearn.metrics import r2_score

# ロギング設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)


# =================================
# CatBoost で多重回帰（MultiRMSE）を行うクラス
# =================================
class CatBoostMultiRegressor:
    def __init__(self, **kwargs):
        """
        Parameters:
            train_df       : pd.DataFrame
            val_df         : pd.DataFrame (検証用)
            test_df        : pd.DataFrame (テスト用)
            target_columns : list[str] or str （例：["target_vol", "target_price"]）
        """
        self.train_df = kwargs.get('train_df')
        self.val_df = kwargs.get('val_df', None)
        self.test_df = kwargs.get('test_df')
        
        # target_columns が単一strかリストかの処理
        target = kwargs.get('target_columns', None)
        if isinstance(target, list):
            self.target_columns = target
        else:
            self.target_columns = [target]

        # デフォルトのハイパーパラメータなど
        self.num_epochs = 1000  # CatBoostの反復回数上限
        self.window_size = 3  # 時系列データのウィンドウサイズ
        self.early_stopping_rounds = 30
        self.do_hyperparam_search = True  # ハイパラ探索を行うかどうか
        self.is_hyperparam_search = False  # ハイパラ探索中かどうかのフラグ
        self.stride = 1  # デフォルト値

        # 学習・評価で記録する値（学習曲線と残差プロット、最終評価R2スコアに必要なもの）
        self.evals_result = {
            'train_loss': [],
            'val_loss': [],
            'test_labels': [],
            'test_preds': [],
            'r2_score': None
        }

        self.best_val_loss = float('inf')
        self.best_hyperparams = {}

        # データ作成（TimeSeriesDataset は RNN用だが、本実装ではフラット化だけに利用）
        self.data_set = TimeSeriesDataset(
            train_df=self.train_df,
            val_df=self.val_df,
            test_df=self.test_df,
            target_columns=self.target_columns,  # マルチタスク
            window_size=self.window_size
            # 初期はstrideは渡さない（後でハイパラ探索結果を反映）
        )

    def run(self, fold: int = None):
        """
        メイン実行。ハイパーパラメータ探索が有効な場合はOptuna実行し、最終的な学習を行う。
        """
        # ハイパラ探索
        if self.do_hyperparam_search:
            self.is_hyperparam_search = True

            def objective(trial: Trial):
                # ハードコードしたGPU設定
                task_type = 'GPU'
                devices = '0:1'
                print(f"Trial {trial.number} using devices {devices}")

                # trial からハイパーパラメータ
                depth = trial.suggest_int('depth', 4, 7)
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
                l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0, step=1)
                bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0)
                window_size = trial.suggest_int('window_size', 2, 10, step=2)
                stride = trial.suggest_int('stride', 30, 120, step=30)  # ← 追加

                # エポック上限は長めにして、early_stopping_roundsで打ち切る
                num_epochs = 1000

                # trial 用に一時的にデータセットを組み直す（window_size, stride が変わるため）
                trial_dataset = TimeSeriesDataset(
                    train_df=self.train_df,
                    val_df=self.val_df,
                    test_df=self.test_df,
                    target_columns=self.target_columns,
                    window_size=window_size,
                    stride=stride  # ← 追加
                )

                # フラット化した学習データ・検証データを取得
                X_train, y_train, X_val, y_val = self._prepare_data_for_catboost(trial_dataset)

                # CatBoostモデル定義
                cat_model = CatBoostRegressor(
                    loss_function='MultiRMSE',
                    boosting_type='Plain',
                    depth=depth,
                    learning_rate=learning_rate,
                    l2_leaf_reg=l2_leaf_reg,
                    bagging_temperature=bagging_temperature,
                    random_seed=42,
                    iterations=num_epochs,
                    task_type=task_type,
                    devices=devices,
                    early_stopping_rounds=self.early_stopping_rounds,
                    use_best_model=True,
                    verbose=False,
                    max_bin=254
                )

                # fit
                train_pool = Pool(data=X_train, label=y_train)
                val_pool = Pool(data=X_val, label=y_val) if (X_val is not None and y_val is not None) else None
                cat_model.fit(train_pool, eval_set=val_pool)

                # 評価指標（val）の取得
                best_score = cat_model.get_best_score()
                val_multi_rmse = best_score["validation"]["MultiRMSE"] if val_pool is not None else float('inf')
                return val_multi_rmse

            # Optuna スタディを作成・実行（探索はシーケンシャルに実行）
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=2)
            )
            study.optimize(objective, n_trials=30, n_jobs=1)

            best_params = study.best_trial.params
            self.best_hyperparams = best_params

            # 得たパラメータを内部に反映（stride も追加）
            self.window_size = best_params['window_size']
            self.depth = best_params['depth']
            self.learning_rate = best_params['learning_rate']
            self.l2_leaf_reg = best_params['l2_leaf_reg']
            self.bagging_temperature = best_params['bagging_temperature']
            self.stride = best_params['stride']  # ← 追加

            # 最新のウィンドウサイズとstrideを反映するため、データセットを再生成
            self.data_set = TimeSeriesDataset(
                train_df=self.train_df,
                val_df=self.val_df,
                test_df=self.test_df,
                target_columns=self.target_columns,
                window_size=self.window_size,
                stride=self.stride  # ← 追加
            )

            self.is_hyperparam_search = False

        # 最終学習・評価（fold=None の場合は全データでの学習）
        result_metrics = self._train_and_evaluate(fold=fold)
        return {
            'evaluation_results': {
                'primary_model': {
                    'val_loss': result_metrics.get("val_loss", None),
                    'r2_score': result_metrics.get("r2_score", None)
                }
            }
        }

    def _prepare_data_for_catboost(self, dataset: TimeSeriesDataset):
        """
        TimeSeriesDataset の (N, window_size, feature_dim) の特徴量と (N, num_target) のラベルを
        (N, window_size * feature_dim) にフラット化して返す。
        """
        X_train = None
        y_train = None
        X_val = None
        y_val = None

        if dataset.train_features is not None:
            train_feat_np = dataset.train_features.numpy()
            N, W, F = train_feat_np.shape
            X_train = train_feat_np.reshape(N, W * F)
            y_train = dataset.train_labels.numpy()

        if dataset.val_features is not None:
            val_feat_np = dataset.val_features.numpy()
            N, W, F = val_feat_np.shape
            X_val = val_feat_np.reshape(N, W * F)
            y_val = dataset.val_labels.numpy()

        return X_train, y_train, X_val, y_val

    def _prepare_test_data_for_catboost(self, dataset: TimeSeriesDataset):
        """
        テスト用データを (N, window_size * feature_dim) にフラット化。
        """
        X_test = None
        y_test = None

        if dataset.test_features is not None:
            test_feat_np = dataset.test_features.numpy()
            N, W, F = test_feat_np.shape
            X_test = test_feat_np.reshape(N, W * F)
            y_test = dataset.test_labels.numpy()

        return X_test, y_test

    def _train_and_evaluate(self, fold=None):
        """
        CatBoostで学習～評価を行う。
        ハイパーパラメータ探索中はチェックポイント保存などは行いません。
        """
        # データセットからフラット化した特徴量・ラベルを取得
        X_train, y_train, X_val, y_val = self._prepare_data_for_catboost(self.data_set)
        X_test, y_test = self._prepare_test_data_for_catboost(self.data_set)

        train_pool = Pool(data=X_train, label=y_train) if X_train is not None else None
        val_pool = Pool(data=X_val, label=y_val) if X_val is not None else None

        # ハイパーパラメータがセットされていない場合はデフォルト値
        depth = getattr(self, 'depth', 6)
        learning_rate = getattr(self, 'learning_rate', 0.05)
        l2_leaf_reg = getattr(self, 'l2_leaf_reg', 3.0)
        bagging_temperature = getattr(self, 'bagging_temperature', 0.5)

        # ハードコードしたGPU設定
        task_type = 'GPU'
        devices = '0:1'
        cat_model = CatBoostRegressor(
            loss_function='MultiRMSE',
            boosting_type='Plain',
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            bagging_temperature=bagging_temperature,
            random_seed=42,
            iterations=self.num_epochs,
            task_type=task_type,
            devices=devices,
            early_stopping_rounds=self.early_stopping_rounds,
            use_best_model=True,
            verbose=False,
            max_bin=254
        )

        if train_pool is not None:
            cat_model.fit(train_pool, eval_set=val_pool)

        # 学習曲線の保存（各反復での損失値）
        evals = cat_model.get_evals_result()
        self.evals_result['train_loss'] = evals.get('learn', {}).get('MultiRMSE', [])
        self.evals_result['val_loss'] = evals.get('validation', {}).get('MultiRMSE', [])

        best_score = cat_model.get_best_score()
        val_loss = best_score["validation"]["MultiRMSE"] if val_pool is not None else float('inf')
        self.best_val_loss = val_loss

        # 追加：特徴量の寄与率（feature importance）の取得と保存
        self.evals_result['feature_importance'] = cat_model.get_feature_importance(train_pool).tolist()

        # テスト予測と評価（各ターゲットごとに評価）
        if X_test is not None:
            test_preds = cat_model.predict(X_test)  # shape: (N, num_targets)
            # 万が一1次元になっている場合は2次元に変換
            if len(test_preds.shape) == 1:
                test_preds = test_preds.reshape(-1, 1)
            if len(y_test.shape) == 1:
                y_test = y_test.reshape(-1, 1)
                
            #ベストハイパーパラメータの表示
            print(f"Best Hyperparameters: {self.best_hyperparams}")

            # 全体のR²スコア（全ターゲットを一括評価）
            overall_r2 = r2_score(y_test, test_preds)
            self.evals_result['r2_score'] = overall_r2
            print(f"[CatBoost] Overall Test R2 Score: {overall_r2:.4f} (fold={fold})")

            # 全体の正解値と予測値を保存
            self.evals_result['test_labels'] = y_test.tolist()
            self.evals_result['test_preds'] = test_preds.tolist()

            # ターゲットごとの評価
            r2_score_per_target = {}
            test_labels_per_target = {}
            test_preds_per_target = {}

            for i, target in enumerate(self.target_columns):
                true_values = y_test[:, i]
                pred_values = test_preds[:, i]
                r2_target = r2_score(true_values, pred_values)
                r2_score_per_target[target] = r2_target
                test_labels_per_target[target] = true_values.tolist()
                test_preds_per_target[target] = pred_values.tolist()
                print(f"[CatBoost] Test R2 Score for {target}: {r2_target:.4f} (fold={fold})")

            self.evals_result['r2_score_per_target'] = r2_score_per_target
            self.evals_result['test_labels_per_target'] = test_labels_per_target
            self.evals_result['test_preds_per_target'] = test_preds_per_target
        
        # ここからモデル保存の処理
        save_dir = "models/catboost"
        os.makedirs(save_dir, exist_ok=True)
        # foldが指定されている場合はファイル名にfold番号を付加、無い場合は最終モデルとして保存
        model_filename = f"catboost_model_fold_{fold}.cbm" if fold is not None else "catboost_model_final.cbm"
        save_path = os.path.join(save_dir, model_filename)
        cat_model.save_model(save_path)
        print(f"モデルを保存しました: {save_path}")
        # ここまで

        return {"val_loss": val_loss, "r2_score": self.evals_result.get('r2_score')}


# ===============================
# メイン処理例（GPU 2枚利用）
# ===============================
if __name__ == '__main__':
    from datetime import datetime, timezone, timedelta

    # 例：train_df, val_df, test_df, target_columns は各自定義してください
    train_df = pd.DataFrame()  # 実データの読み込み処理を記述
    val_df = pd.DataFrame()
    test_df = pd.DataFrame()
    target_columns = ["target_price", "target_vol"]  # マルチターゲットの例

    model_instance = CatBoostMultiRegressor(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_columns=target_columns
    )

    # ハイパラ探索を有効にする場合
    model_instance.do_hyperparam_search = True

    # ハイパラ探索はシンプルにシーケンシャル実行
    results = model_instance.run()
    print("最終評価結果:", results)


# =================================
# 以下、foldごとの学習・評価処理など
# =================================

def load_data(file_path: Path) -> pd.DataFrame:
    logger.info(f"データ読み込み: {file_path}")
    data = pd.read_pickle(file_path)
    logger.info(f"データ読み込み完了: {file_path}")
    object_columns = data.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        logger.warning(f"object型の列が存在: {object_columns}")
    return data

def get_file_paths(data_dir: Path, filename: str, fold: int) -> (Path, Path):
    train_file = data_dir / f"{filename}_fold{fold}_train.pkl"
    test_file = data_dir / f"{filename}_fold{fold}_test.pkl"
    return train_file, test_file

def train_model_for_fold(fold: int, config: dict) -> dict:
    logger.info(f"フォールド {fold} の学習開始")
    data_dir = config["data_dir"]
    filename = config["filename"]
    train_file, test_file = get_file_paths(data_dir, filename, fold)
    
    train_df = load_data(train_file)
    test_df = load_data(test_file)
    
    train_df = train_df.drop(columns=config["exclude_target"])
    test_df = test_df.drop(columns=config["exclude_target"])
    
    # device引数はハードコードしているため不要
    model = CatBoostMultiRegressor(
        train_df=train_df,
        test_df=test_df,
        target_columns=config["target_columns"]
    )

    result = model.run(fold)
    logger.info(f"フォールド {fold} の学習完了。結果: {result}")
    return {
        "fold": fold,
        "result": result,
        "evals_result": model.evals_result,
        "best_hyperparams": model.best_hyperparams
    }

def aggregate_results(results: list):
    if not results:
        logger.warning("集約する結果がありません。")
        return
    total_loss = 0
    total_r2 = 0
    count = 0
    print("\n==== フォールド別結果 ====")
    for res in results:
        fold = res["fold"]
        performance = res["result"]["evaluation_results"]["primary_model"]
        val_loss = performance.get("val_loss", None)
        r2 = performance.get("r2_score", None)
        print(f"フォールド {fold}: 検証損失 = {val_loss:.4f},  R2 Score = {r2:.4f}")
        total_loss += val_loss
        total_r2 += r2
        count += 1
    avg_loss = total_loss / count
    avg_r2 = total_r2 / count
    print("-----------------------------")
    print(f"全体平均検証損失: {avg_loss:.4f}")
    print(f"全体平均R2 Score: {avg_r2:.4f}\n")

def train_model_on_full_data(config: dict, final_hyperparams: dict) -> dict:
    logger.info("全データを用いた最終学習を開始")
    data_dir = config["data_dir"]
    filename = config["filename"]

    # 全データの訓練ファイル
    full_file = data_dir / f"{filename}_train.pkl"
    # テスト用ファイル
    test_file = data_dir / f"{filename}_test.pkl"

    full_df = load_data(full_file)
    test_df = load_data(test_file)

    full_df = full_df.drop(columns=config["exclude_target"])
    test_df = test_df.drop(columns=config["exclude_target"])

    # device引数は不要
    model = CatBoostMultiRegressor(
        train_df=full_df,
        test_df=test_df,
        target_columns=config["target_columns"]
    )

    # ハイパラ探索で得たパラメータの反映
    if final_hyperparams:
        model.window_size         = final_hyperparams.get("window_size", model.window_size)
        model.depth               = final_hyperparams.get("depth", getattr(model, 'depth', 6))
        model.learning_rate       = final_hyperparams.get("learning_rate", getattr(model, 'learning_rate', 0.03))
        model.l2_leaf_reg         = final_hyperparams.get("l2_leaf_reg", getattr(model, 'l2_leaf_reg', 3.0))
        model.bagging_temperature = final_hyperparams.get("bagging_temperature", getattr(model, 'bagging_temperature', 1.0))
        model.stride              = final_hyperparams.get("stride", 30)  # ← 追加
    else:
        logger.warning("best_hyperparams が空です。デフォルトのハイパーパラメータを使用します。")

    # 最新のウィンドウサイズとstrideを反映するため、データセットを再生成
    model.data_set = TimeSeriesDataset(
        train_df=full_df,
        val_df=None,
        test_df=test_df,
        target_columns=config["target_columns"],
        window_size=model.window_size,
        stride=model.stride  # ← 追加
    )

    # 最終学習実行（fold=None で再学習）
    final_result_metrics = model.run()
    final_result_metrics["evals_result"] = model.evals_result

    # テスト結果（予測値・正解値を含む）を保存
    results_save_dir = Path("storage/predictions")
    results_save_dir.mkdir(parents=True, exist_ok=True)
    target_str = "_".join(config["target_columns"])
    results_save_path = results_save_dir / f"catboost_test_results_{target_str}.pkl"
    with open(results_save_path, "wb") as f:
        pickle.dump(final_result_metrics, f)
    print(f"最終テスト結果を {results_save_path} に保存しました")

    logger.info("全データを用いた最終学習完了")
    return {
        "result": final_result_metrics,
        "evals_result": model.evals_result,
        "best_hyperparams": model.best_hyperparams
    }

def run_folds(config: dict):
    results = []
    num_folds = config.get("num_folds", 5)
    for fold in range(num_folds):
        res = train_model_for_fold(fold, config)
        results.append(res)
    aggregate_results(results)
    logger.info("全フォールドの学習完了。")

    # CVの結果から、テスト損失（ここでは検証損失）が最小のフォールドのハイパーパラメータを採用
    best_fold = min(results, key=lambda x: x["result"]["evaluation_results"]["primary_model"]["val_loss"])
    final_hyperparams = best_fold["best_hyperparams"]
    print("選択された最良ハイパーパラメータ:", final_hyperparams)

    # 全データで再学習
    final_training_result = train_model_on_full_data(config, final_hyperparams)

    plot = {
        "fold_results": results,
        "final_result": final_training_result
    }
    plot_final_training_results_multitask(plot)
