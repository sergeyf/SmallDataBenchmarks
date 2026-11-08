"""
How good is LightGBM with no tuning? Defaults, except 1000 trees to match CatBoost's default iteration count.
"""

import lightgbm as lgb

from utils import N_JOBS, fit_then_predict, run_benchmark

N_ESTIMATORS = 1000  # CatBoost's default; LightGBM's own default is 100


def fit(X_train, y_train, random_state):
    # objective and metric match what LightGBM picks itself; tree_learner only changes parallelism.
    binary = len(set(y_train)) == 2
    model = lgb.LGBMClassifier(
        objective="binary" if binary else "multiclass",
        n_estimators=N_ESTIMATORS,
        metric="auc" if binary else "auc_mu",
        verbose=-1,
        tree_learner="feature",
        random_state=random_state,
        n_jobs=N_JOBS,
    )
    return model.fit(X_train, y_train)


if __name__ == "__main__":
    run_benchmark(f"lightgbm_default_{N_ESTIMATORS}", fit_then_predict(fit))
