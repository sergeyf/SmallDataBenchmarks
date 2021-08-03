"""
How does LightGBM compare?
"""

import time
import pickle
import numpy as np
import lightgbm as lgb
from functools import partial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll import scope
from utils import load_data

N_JOBS = 4 * 4 * 9
N_ITER = 50  # budget for hyperparam search
N_STARTUP_JOBS = 20  # hyperopt does a bunch of random evals first

HYPEROPT_SPACE = {
    "learning_rate": hp.choice("learning_rate", [0.1, 0.05, 0.01, 0.005, 0.001]),
    "num_leaves": scope.int(2 ** hp.quniform("num_leaves", 2, 7, 1)),
    "colsample_bytree": hp.quniform("colsample_bytree", 0.4, 1, 0.1),
    "subsample": hp.quniform("subsample", 0.4, 1, 0.1),
    "min_child_samples": scope.int(2 ** hp.quniform("min_child_samples", 0, 7, 1)),
    "min_child_weight": 10 ** hp.quniform("min_child_weight", -6, 0, 1),
    "reg_alpha": hp.choice("reg_alpha", [0, 10 ** hp.quniform("reg_alpha_pos", -6, 1, 1)]),
    "reg_lambda": hp.choice("reg_lambda", [0, 10 ** hp.quniform("reg_lambda_pos", -6, 1, 1)]),
    "max_depth": scope.int(hp.choice("max_depth", [-1, 2 ** hp.quniform("max_depth_pos", 1, 4, 1)])),
}


def define_and_evaluate_lightgbm_pipeline(X, y, random_state=0):
    binary = len(set(y)) == 2
    if binary:
        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            metric="auc",
            verbose=-1,
            tree_learner="feature",
            random_state=random_state,
            silent=True,
        )
    else:
        lgb_model = lgb.LGBMClassifier(
            objective="multiclass",
            n_estimators=500,
            metric="auc_mu",
            verbose=-1,
            tree_learner="feature",
            random_state=random_state,
            silent=True,
        )
    nested_scores = []
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    for train_inds, test_inds in outer_cv.split(X, y):
        X_train, y_train = X[train_inds, :], y[train_inds]
        X_test, y_test = X[test_inds, :], y[test_inds]

        def obj(params):
            lgb_model.set_params(**params)
            inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
            scores = cross_val_score(
                lgb_model, X=X_train, y=y_train, cv=inner_cv, scoring="roc_auc_ovr_weighted", n_jobs=N_JOBS
            )
            return -np.mean(scores)

        trials = Trials()
        _ = fmin(
            fn=obj,
            space=HYPEROPT_SPACE,
            algo=partial(tpe.suggest, n_startup_jobs=N_STARTUP_JOBS),
            max_evals=N_ITER,
            trials=trials,
            rstate=np.random.RandomState(random_state),
        )
        # hyperopt has some problems with hp.choice so we need to do this:
        best_params = space_eval(HYPEROPT_SPACE, trials.argmin)
        lgb_model.set_params(**best_params)
        lgb_model.fit(X_train, y_train)
        y_pred = lgb_model.predict_proba(X_test)
        # same as roc_auc_ovr_weighted
        if binary:
            score = roc_auc_score(y_test, y_pred[:, 1], average="weighted", multi_class="ovr")
        else:
            score = roc_auc_score(y_test, y_pred, average="weighted", multi_class="ovr")
        nested_scores.append(score)
    return nested_scores


# run model on all datasets
with open("results/01_compare_baseline_models.pickle", "rb") as f:
    _, _, _, evaluated_datasets, _ = pickle.load(f)


results = []
times = []
for i, dataset_name in enumerate(evaluated_datasets):
    X, y = load_data(dataset_name)
    np.random.seed(0)
    if len(y) > 10000:
        # subset to 10000 if too large
        random_idx = np.random.choice(len(y), 10000, replace=False)
        X = X[random_idx, :]
        y = y[random_idx]
    print("starting:", dataset_name, X.shape)
    start = time.time()
    nested_scores = define_and_evaluate_lightgbm_pipeline(X, y)
    results.append(nested_scores)
    elapsed = time.time() - start
    times.append(elapsed)
    print("done. elapsed:", elapsed)

#
results = np.array(results)
times = np.array(times)

# save everything to disk so we can make plots elsewhere
with open(f"results/02a_lightgbm_n_iter_{N_ITER}.pickle", "wb") as f:
    pickle.dump((results, times), f)
