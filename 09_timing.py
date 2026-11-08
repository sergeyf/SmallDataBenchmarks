"""
How long does each model take to train and to predict, on CPU and (where supported) on GPU?

Times fit and predict_proba separately on one outer fold of the 30 datasets in timing_datasets.txt.
Each model and device gets one untimed warm-up fit first, so one-time costs (CUDA start-up, loading
pretrained weights) are not counted. Run it on an otherwise idle machine. Results go to results/2026/timing.csv
and resume like the other scripts.
"""

import argparse
import csv
import importlib.util
import os
import time
from functools import partial

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from utils import N_JOBS, RESULTS_DIR, load_data, subsample, weighted_auroc

os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")  # TabPFN refuses >5000 rows on CPU by default
torch.set_num_threads(N_JOBS)  # TabPFN and TabICL on CPU


def load(script: str):
    spec = importlib.util.spec_from_file_location(script.removesuffix(".py"), script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timing_datasets() -> list[str]:
    """Return the timing datasets, smallest first; the first one is used for warm-up.

    They were picked evenly by row count from the filtered benchmark datasets, minus autoUniv-au4-2500,
    where linear SVC runs for hours. The list is frozen because the filtered set depends on which models
    have finished, so picking again would time different datasets on a rerun or resume.
    """
    with open("timing_datasets.txt") as f:
        return [line.strip() for line in f if line.strip()]


def runs():
    baselines = load("01_compare_baseline_models.py").FITS
    lightgbm, autogluon = load("02_lightgbm.py"), load("03_autogluon.py")
    lightgbm_default = load("04_lightgbm_default.py")
    catboost, tabpfn, tabicl = load("05_catboost.py"), load("06_tabpfn.py"), load("07_tabicl.py")
    yield "Logistic Regression", "cpu", baselines["logistic_regression"]
    yield "SVC", "cpu", baselines["svc"]
    yield "Random Forest", "cpu", baselines["random_forest"]
    yield "LightGBM (hyperopt, 50 trials)", "cpu", lightgbm.fit  # the pip build has no GPU support
    yield "LightGBM (default, 1000 trees)", "cpu", lightgbm_default.fit
    for device in ["cpu", "gpu"]:
        yield "CatBoost (default)", device, partial(catboost.fit, device=device)
        yield "AutoGluon 1.6 (60s)", device, partial(autogluon.fit, device=device)
        yield "TabPFN-3.5", device, partial(tabpfn.fit, device=device)
        yield "TabICLv2", device, partial(tabicl.fit, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    # Run devices separately so the GPU part can wait until the GPU is free; both resume from timing.csv.
    parser.add_argument("--device", choices=["cpu", "gpu", "all"], default="all")
    device_filter = parser.parse_args().device
    path = f"{RESULTS_DIR}/timing.csv"
    done = set()
    # Check the file, not `done`: a run that crashed before its first row leaves a header-only file.
    write_header = not (os.path.exists(path) and os.path.getsize(path) > 0)
    if not write_header:
        done = set(pd.read_csv(path)[["model", "device", "dataset"]].itertuples(index=False, name=None))
    datasets = timing_datasets()
    folds = {}
    for name in datasets:
        X, y = subsample(*load_data(name))
        train, test = next(StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(X, y))
        folds[name] = (X[train], y[train], X[test], y[test])
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(
                ["model", "device", "dataset", "n_train", "n_test", "fit_seconds", "predict_seconds", "auroc"]
            )
        for model, device, fit in runs():
            if device_filter != "all" and device != device_filter:
                continue
            todo = [d for d in datasets if (model, device, d) not in done]
            if not todo:
                continue
            X_train, y_train, X_test, _ = folds[datasets[0]]
            fit(X_train, y_train, 0).predict_proba(X_test)  # warm-up, not timed
            for dataset in todo:
                X_train, y_train, X_test, y_test = folds[dataset]
                start = time.perf_counter()
                fitted = fit(X_train, y_train, 0)
                fit_seconds = time.perf_counter() - start
                start = time.perf_counter()
                proba = fitted.predict_proba(X_test)
                predict_seconds = time.perf_counter() - start
                auroc = weighted_auroc(y_test, proba)
                writer.writerow(
                    [model, device, dataset, len(y_train), len(y_test), fit_seconds, predict_seconds, auroc]
                )
                f.flush()
                print(
                    f"{model} [{device}] {dataset}: fit {fit_seconds:.1f}s, predict {predict_seconds:.2f}s", flush=True
                )


if __name__ == "__main__":
    main()
