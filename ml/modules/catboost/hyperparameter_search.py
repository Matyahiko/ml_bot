import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from modules.catboost_model import CatBoostMultiRegressor  # モジュールの配置に合わせてインポートしてください

def hyperparameter_search(train_df, val_df, test_df, target_columns, n_trials=30, seed=42):
    """
    指定されたデータに対して、Optuna を用いてハイパーパラメータ探索を行い、
    最良のパラメータを dict 形式で返します。
    """
    def objective(trial):
        task_type = 'GPU'
        devices = '0:1'
        print(f"Trial {trial.number} using devices {devices}")

        # trial から各パラメータをサジェスト
        depth = trial.suggest_int('depth', 4, 7)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-3, 1e-1)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 1.0, 10.0, step=1)
        bagging_temperature = trial.suggest_float('bagging_temperature', 0.0, 1.0)
        window_size = trial.suggest_int('window_size', 2, 10, step=2)
        stride = trial.suggest_int('stride', 30, 120, step=30)

        num_epochs = 1000
        hyperparams = {
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "bagging_temperature": bagging_temperature
        }
        # 各 trial ごとに一時的なモデルを生成
        model = CatBoostMultiRegressor(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            target_columns=target_columns,
            window_size=window_size,
            stride=stride,
            num_epochs=num_epochs,
            early_stopping_rounds=30,
            hyperparams=hyperparams
        )
        result = model.train_and_evaluate()
        val_loss = result.get("val_loss", float('inf'))
        return val_loss

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=2)
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    best_params = study.best_trial.params
    # window_size と stride も含める
    best_params['window_size'] = study.best_trial.params.get('window_size')
    best_params['stride'] = study.best_trial.params.get('stride')
    return best_params
