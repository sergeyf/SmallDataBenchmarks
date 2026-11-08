"""
How does LightGBM compare? Tuned with hyperopt on inner 4-fold CV.
"""

from functools import partial

import lightgbm as lgb
import numpy as np
from hyperopt import Trials, fmin, hp, space_eval, tpe
from hyperopt.pyll import scope
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import N_FOLDS, N_JOBS, fit_then_predict, run_benchmark

N_ITER = 50  # budget for hyperparam search
N_STARTUP_JOBS = 20  # hyperopt does a bunch of random evals first

HYPEROPT_SPACE = {
    "learning_rate": hp.choice("learning_rate", [0.1, 0.05, 0.01, 0.005, 0.001]),
    "num_leaves": scope.int(2 ** hp.quniform("num_leaves", 2, 7, 1)),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.4, 1, 0.1),
    # Inactive: LightGBM only bags rows when subsample_freq > 0. Kept as in 2020 so scores stay comparable.
    "subsample": hp.quniform("subsample", 0.4, 1, 0.1),
    "min_child_samples": scope.int(2 ** hp.quniform("min_child_samples", 0, 7, 1)),
    "min_child_weight": 10 ** hp.quniform("min_child_weight", -6, 0, 1),
    "reg_alpha": hp.choice("reg_alpha", [0, 10 ** hp.quniform("reg_alpha_pos", -6, 1, 1)]),
    "reg_lambda": hp.choice("reg_lambda", [0, 10 ** hp.quniform("reg_lambda_pos", -6, 1, 1)]),
    "max_depth": scope.int(hp.choice("max_depth", [-1, 2 ** hp.quniform("max_depth_pos", 1, 4, 1)])),
}


def fit(X_train, y_train, random_state):
    # Same settings as the 2020 run; `silent` is gone from LightGBM 4 and verbose=-1 replaces it.
    binary = len(set(y_train)) == 2
    model = lgb.LGBMClassifier(
        objective="binary" if binary else "multiclass",
        n_estimators=500,
        metric="auc" if binary else "auc_mu",
        verbose=-1,
        tree_learner="feature",
        random_state=random_state,
        n_jobs=N_JOBS // N_FOLDS,  # threads per model; the inner folds run in parallel
    )
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)

    def objective(params):
        model.set_params(**params)
        scores = cross_val_score(model, X_train, y_train, cv=inner_cv, scoring="roc_auc_ovr_weighted", n_jobs=N_FOLDS)
        return -np.mean(scores)

    trials = Trials()
    fmin(
        fn=objective,
        space=HYPEROPT_SPACE,
        algo=partial(tpe.suggest, n_startup_jobs=N_STARTUP_JOBS),
        max_evals=N_ITER,
        trials=trials,
        rstate=np.random.default_rng(random_state),
        show_progressbar=False,
    )
    # hyperopt returns indices for hp.choice, so map them back to values.
    model.set_params(**space_eval(HYPEROPT_SPACE, trials.argmin))
    return model.fit(X_train, y_train)


if __name__ == "__main__":
    run_benchmark(f"lightgbm_hyperopt_{N_ITER}", fit_then_predict(fit))
