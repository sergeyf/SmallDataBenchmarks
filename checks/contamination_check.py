"""Check whether TabPFN-3.5 and TabICLv2 have memorized the benchmark datasets.

Test "shuffled": fit on fold 0 with shuffled training labels (three shuffles), then score against the true
test labels. A model that learns from the given rows scores about 0.5; one that memorized the dataset scores higher.
Test "k=10"/"k=20": fit on k random training rows (at least one per class), three seeds each.
Memorization shows as a score close to the "full" score while the controls collapse.
"full" and "shuffled" use at most CAP training rows: full-size TabPFN fits take minutes each on big datasets,
and 1000 rows is plenty to show memorization. CatBoost and a default random forest are the controls.

Run from the repo root: uv run python -m checks.contamination_check [--datasets a b]
The post's run predates nursery-4class joining the filtered set, so it covers 105 of the 106 datasets.
Writes results/2026/contamination.csv and resumes from it.
"""

import argparse
import csv
import importlib.util
import os
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

import figures
from utils import N_JOBS, RESULTS_DIR, load_data, subsample, weighted_auroc

PATH = f"{RESULTS_DIR}/contamination.csv"
CAP = 1000
FEW_ROWS = [10, 20]
SEEDS = [0, 1, 2]


def load(script: str):
    spec = importlib.util.spec_from_file_location(script.removesuffix(".py"), script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_forest(X_train, y_train, random_state):
    # Plain defaults: the 2020 grid search needs more rows than the few-row test has.
    return RandomForestClassifier(n_estimators=500, n_jobs=N_JOBS, random_state=random_state).fit(X_train, y_train)


def models():
    tabpfn, tabicl, catboost = load("06_tabpfn.py"), load("07_tabicl.py"), load("05_catboost.py")
    yield "TabPFN-3.5", partial(tabpfn.fit, device="gpu")
    yield "TabICLv2", partial(tabicl.fit, device="gpu")
    yield "CatBoost (default)", catboost.fit
    yield "Random Forest (default)", random_forest


def few_rows(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick k row indices with every class present, so predict_proba columns stay 0..n_classes-1."""
    first = np.array([rng.choice(np.flatnonzero(y == c)) for c in np.unique(y)])
    rest = rng.choice(np.setdiff1d(np.arange(len(y)), first), k - len(first), replace=False)
    return np.concatenate([first, rest])


def tasks(y_train: np.ndarray):
    """Yield (test, seed, train row indices, training labels) for one dataset."""
    base = np.arange(len(y_train))
    if len(base) > CAP:
        base = few_rows(y_train, CAP, np.random.default_rng(0))
    yield "full", 0, base, y_train[base]
    # Several shuffles: on tiny datasets one shuffle can leave real signal by chance.
    for seed in SEEDS:
        yield "shuffled", seed, base, np.random.default_rng(seed).permutation(y_train[base])
    for k in FEW_ROWS:
        if len(np.unique(y_train)) > k // 2:  # too few rows per class to mean anything
            continue
        for seed in SEEDS:
            rows = few_rows(y_train, k, np.random.default_rng(seed))
            yield f"k={k}", seed, rows, y_train[rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", help="default: the filtered benchmark datasets")
    names = parser.parse_args().datasets
    if not names:
        names = sorted(set(figures.filter_datasets(figures.drop_unfinished(figures.load_results())).dataset))
    folds = {}
    for name in names:
        X, y = subsample(*load_data(name))
        train, test = next(StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(X, y))
        folds[name] = (X[train], y[train], X[test], y[test])
    done = set()
    if os.path.exists(PATH) and os.path.getsize(PATH) > 0:
        done = set(pd.read_csv(PATH)[["model", "dataset", "test", "seed"]].itertuples(index=False, name=None))
    with open(PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not done:
            writer.writerow(["model", "dataset", "test", "seed", "n_train", "auroc"])
        for model, fit in models():
            for name in names:
                X_train, y_train, X_test, y_test = folds[name]
                for test, seed, rows, labels in tasks(y_train):
                    if (model, name, test, seed) in done:
                        continue
                    proba = fit(X_train[rows], labels, seed).predict_proba(X_test)
                    auroc = weighted_auroc(y_test, proba)
                    writer.writerow([model, name, test, seed, len(rows), auroc])
                    f.flush()
                    print(f"{model} {name} {test} seed={seed}: auroc={auroc:.4f}", flush=True)


if __name__ == "__main__":
    main()
