"""
How does LightGBM compare?
"""

import time
import pickle
import numpy as np
import lightgbm as lgb
from functools import partial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from hpbandster_sklearn import HpBandSterSearchCV
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from utils import load_data

N_JOBS = 32
N_ITER = 50  # budget for hyperparam search

HPBANDSTER_SPACE = CS.ConfigurationSpace(seed=0)
learning_rate = CSH.CategoricalHyperparameter(name="learning_rate", choices=[0.1, 0.05, 0.01, 0.005, 0.001])
num_leaves = CSH.UniformIntegerHyperparameter(name="num_leaves", lower=2 ** 2, upper=2 ** 7, log=True)
colsample_bytree = CSH.UniformFloatHyperparameter(name="colsample_bytree", lower=0.4, upper=1, q=0.1)
subsample = CSH.UniformFloatHyperparameter(name="subsample", lower=0.4, upper=1, q=0.1)
min_child_samples = CSH.UniformIntegerHyperparameter(name="min_child_samples", lower=2 ** 0, upper=2 ** 7, log=True)
min_child_weight = CSH.UniformFloatHyperparameter(name="min_child_weight", lower=10 ** -6, upper=10 ** 0, log=True)
reg_alpha = CSH.UniformFloatHyperparameter(name="reg_alpha", lower=10 ** -6, upper=10 ** 0, log=True)
reg_lambda = CSH.UniformFloatHyperparameter(name="reg_lambda", lower=10 ** -6, upper=10 ** 0, log=True)
max_depth = CSH.UniformIntegerHyperparameter(name="max_depth", lower=2 ** 1, upper=2 ** 5, log=True)
HPBANDSTER_SPACE.add_hyperparameters(
    [
        learning_rate,
        num_leaves,
        colsample_bytree,
        subsample,
        min_child_samples,
        min_child_weight,
        reg_alpha,
        reg_lambda,
        max_depth,
    ]
)


def define_and_evaluate_hyperbandster_pipeline(X, y, random_state=0):
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

        search = HpBandSterSearchCV(
            lgb_model,
            HPBANDSTER_SPACE,
            resource_name="n_estimators",
            resource_type=int,
            n_iter=N_ITER,
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state),
            scoring="roc_auc_ovr_weighted",
            min_budget=50,
            max_budget=500,
            n_jobs=N_JOBS,
            random_state=random_state,
            refit=True,
        ).fit(X_train, y_train)

        y_pred = search.best_estimator_.predict_proba(X_test)
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
    nested_scores = define_and_evaluate_hyperbandster_pipeline(X, y)
    results.append(nested_scores)
    elapsed = time.time() - start
    times.append(elapsed)
    print("done. elapsed:", elapsed)

#
results = np.array(results)
times = np.array(times)

# save everything to disk so we can make plots elsewhere
with open(f"results/02b_lightgbm_hpbandster_n_iter_{N_ITER}.pickle", "wb") as f:
    pickle.dump((results, times), f)
