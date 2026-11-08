"""Shared data loading and the nested cross-validation loop used by every benchmark script."""

import csv
import os
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RESULTS_DIR = "results/2026"
MAX_ROWS = 10000
N_FOLDS = 4
N_JOBS = 20  # CPU threads to use; leaves a few cores free on the 24-core machine

# A model takes (X_train, y_train, X_test, random_state) and returns class probabilities for X_test.
FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def fit_then_predict(fit: Callable) -> FitPredict:
    """Build a fit_predict function from fit(X_train, y_train, random_state) -> fitted model with predict_proba."""

    def fit_predict(X_train, y_train, X_test, random_state):
        return fit(X_train, y_train, random_state).predict_proba(X_test)

    return fit_predict


def _to_numeric_if_possible(col: pd.Series) -> pd.Series:
    """Decode ARFF byte strings and convert to numbers when every value parses."""
    if col.dtype != object:
        return col
    col = col.str.decode("utf-8")
    try:
        return pd.to_numeric(col)
    except ValueError:
        return col


def load_data(data_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one UCI++ dataset as a one-hot float matrix and integer labels.

    Args:
        data_name: File stem under `datasets/`.

    Returns:
        Features of shape (n, d) and labels numbered in order of first appearance.
    """
    data, _ = arff.loadarff(f"datasets/{data_name}.arff")
    df = pd.DataFrame(data).apply(_to_numeric_if_possible)
    y = pd.factorize(df.pop("Class"))[0]
    X = pd.get_dummies(df).to_numpy(dtype=float)
    return X, y


def dataset_names() -> list[str]:
    """Return the 142 datasets evaluated in the 2020 run, in the same order.

    The list is frozen in `datasets.txt` because the 2020 loader skipped four all-categorical datasets.
    """
    with open("datasets.txt") as f:
        return [line.strip() for line in f if line.strip()]


def subsample(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cap a dataset at MAX_ROWS rows with the same seed as the 2020 run."""
    if len(y) <= MAX_ROWS:
        return X, y
    np.random.seed(0)
    idx = np.random.choice(len(y), MAX_ROWS, replace=False)
    return X[idx], y[idx]


def weighted_auroc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Score with weighted one-vs-rest AUROC, the metric used throughout."""
    if proba.shape[1] == 2:
        return roc_auc_score(y_true, proba[:, 1])
    return roc_auc_score(y_true, proba, average="weighted", multi_class="ovr")


def _has_rows(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _done_datasets(path: str) -> set[str]:
    if not _has_rows(path):
        return set()
    return set(pd.read_csv(path)["dataset"])


def run_benchmark(
    name: str,
    fit_predict: FitPredict,
    skip: Callable[[np.ndarray, np.ndarray], str | None] | None = None,
    random_state: int = 0,
) -> None:
    """Run one model over every dataset with stratified outer folds and append scores to a CSV.

    Datasets already in the CSV are skipped, so an interrupted run resumes where it stopped.

    Args:
        name: Model name; results go to `results/2026/<name>.csv`.
        fit_predict: Trains on one outer training fold and returns test-fold probabilities.
        skip: Optional check that returns a reason when the model cannot run on a dataset.
        random_state: Seed for the outer folds.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = f"{RESULTS_DIR}/{name}.csv"
    done = _done_datasets(path)
    names = dataset_names()
    write_header = not _has_rows(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "fold", "auroc", "seconds", "note"])
        for i, dataset in enumerate(names):
            if dataset in done:
                continue
            X, y = subsample(*load_data(dataset))
            reason = skip(X, y) if skip else None
            if reason:
                print(f"[{i + 1}/{len(names)}] {dataset} {X.shape}: skipped ({reason})", flush=True)
                writer.writerow([dataset, -1, "", "", f"skipped: {reason}"])
                f.flush()
                continue
            outer_cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
            rows = []
            for fold, (train, test) in enumerate(outer_cv.split(X, y)):
                start = time.time()
                proba = fit_predict(X[train], y[train], X[test], random_state)
                rows.append([dataset, fold, weighted_auroc(y[test], proba), time.time() - start, ""])
            writer.writerows(rows)
            f.flush()
            mean_auroc = np.mean([r[2] for r in rows])
            seconds = sum(r[3] for r in rows)
            print(f"[{i + 1}/{len(names)}] {dataset} {X.shape}: auroc={mean_auroc:.4f} in {seconds:.0f}s", flush=True)
