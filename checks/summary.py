"""Print the numbers behind the post's checks: 2020 baselines, AutoML budgets, and LightGBM tuned defaults.

Reads only results files, so it runs in seconds. The memorization check has its own contamination_summary.py.
Run from the repo root: uv run python -m checks.summary
"""

import pickle

import numpy as np
import pandas as pd

import figures
from utils import RESULTS_DIR

SMALL = 250  # rows; the post's "tiny data" cutoff


def benchmark() -> tuple[pd.DataFrame, pd.Series]:
    """Return the filtered per-dataset results and each dataset's row count, as figures.py uses them."""
    results = figures.filter_datasets(figures.drop_unfinished(figures.load_results()))
    database = pd.read_json("database.json").T
    nrow = np.minimum(10000, database["nrow"].astype(int))
    return results, nrow


def per_dataset(path: str) -> pd.Series:
    """Mean AUROC per dataset from a run_benchmark CSV."""
    return pd.read_csv(path).groupby("dataset").auroc.mean()


def baselines_vs_2020(datasets: list[str]) -> None:
    print("== 2020 baselines rerun with today's libraries ==")
    with open("results/01_compare_baseline_models.pickle", "rb") as f:
        svc, logistic, forest, names, _ = pickle.load(f)
    old = {"svc": svc, "logistic_regression": logistic, "random_forest": forest}
    for key, scores in old.items():
        then = pd.Series(scores.mean(axis=1), index=names)
        now = per_dataset(f"{RESULTS_DIR}/{key}.csv").reindex(names)
        diff = (now - then).round(4)
        for label, keep in [("all 142", names), (f"filtered {len(datasets)}", datasets)]:
            d = diff[keep]
            changed = d[d != 0].sort_values()
            print(
                f"{key:20} {label:13}: {len(changed)} differ at 4 decimals; "
                f"mean then {then[keep].mean():.4f}, now {now[keep].mean():.4f}"
            )
            if label.startswith("filtered") and len(changed):
                print("    " + ", ".join(f"{name} {delta:+.4f}" for name, delta in changed.items()))


def autogluon_budget(results: pd.DataFrame) -> None:
    print("\n== AutoGluon: 60 s vs 300 s on its 10 biggest losses ==")
    check = pd.read_csv(f"{RESULTS_DIR}/autogluon_budget_check.csv")
    table = check.groupby(["dataset", "budget"]).auroc.mean().unstack("budget")
    scores = results.pivot(index="dataset", columns="model", values="mean_auroc")
    table["main run (60 s)"] = scores["AutoGluon 1.6 (60s)"]
    table["2020 (300 s)"] = scores["AutoGluon (2020, 300s)"]
    print(table.round(3).to_string())
    print(f"300 s minus 2020, mean over the 10: {(table[300] - table['2020 (300 s)']).mean():+.4f}")
    # The 60 s rerun repeats the main run's folds, so fold-level gaps show run-to-run noise.
    main = pd.read_csv(f"{RESULTS_DIR}/autogluon_1_6_sec_60.csv")
    rerun = check[check.budget == 60].merge(main, on=["dataset", "fold"], suffixes=("_rerun", "_main"))
    swing = (rerun.auroc_rerun - rerun.auroc_main).abs().groupby(rerun.dataset).max()
    print("largest single-fold change, 60 s rerun vs main run:")
    print(swing.sort_values(ascending=False).round(3).to_string())


def mljar_validation(results: pd.DataFrame, nrow: pd.Series) -> None:
    print("\n== MLJAR: forced 10-fold CV vs MLJAR's own (90/10 split) choice, 60 s ==")
    datasets = results.dataset.unique()
    split = per_dataset(f"{RESULTS_DIR}/mljar_1_3_sec_60_split_validation.csv")[datasets]
    kfold = per_dataset(f"{RESULTS_DIR}/mljar_1_3_sec_60.csv")[datasets]
    small = nrow[datasets] < SMALL
    print(f"mean, all {len(datasets)}: split {split.mean():.3f}, 10-fold {kfold.mean():.3f}")
    print(f"mean, {small.sum()} under {SMALL} rows: split {split[small].mean():.3f}, 10-fold {kfold[small].mean():.3f}")
    delta = (kfold - split).sort_values()
    print("largest drops with 10-fold: " + ", ".join(f"{d} {delta[d]:+.3f} ({nrow[d]} rows)" for d in delta.index[:3]))
    scores = results.pivot(index="dataset", columns="model", values="mean_auroc")
    loss = (split - scores["MLJAR (2020, 300s)"]).sort_values()
    print("split run vs MLJAR 2020, largest losses: " + ", ".join(f"{d} {loss[d]:+.3f}" for d in loss.index[:3]))

    check = pd.read_csv(f"{RESULTS_DIR}/mljar_budget_check.csv")
    print("split validation, 60 s vs 300 s:")
    print(check.groupby(["dataset", "budget"]).auroc.mean().unstack("budget").round(3).to_string())
    boards = pd.read_csv(f"{RESULTS_DIR}/mljar_budget_leaderboards.csv")
    tree = boards[boards.name == "1_DecisionTree"].train_time
    print(f"first decision tree train_time: {tree.min():.2f}-{tree.max():.2f} s (k-fold needs under SEC/250 s)")


def lightgbm_tuned_defaults(results: pd.DataFrame) -> None:
    print("\n== LightGBM tuned defaults (Better by Default, via pytabkit) ==")
    datasets = results.dataset.unique()
    tuned = per_dataset(f"{RESULTS_DIR}/lightgbm_tuned_defaults.csv").reindex(datasets)
    scores = results.pivot(index="dataset", columns="model", values="mean_auroc")
    print(f"{tuned.notna().sum()} of {len(datasets)} datasets; mean AUROC {tuned.mean():.3f}")
    for model in ["LightGBM (default, 1000 trees)", "LightGBM (hyperopt, 50 trials)"]:
        print(f"{model}: {scores[model].mean():.3f}")


def main() -> None:
    results, nrow = benchmark()
    baselines_vs_2020(sorted(results.dataset.unique()))
    autogluon_budget(results)
    mljar_validation(results, nrow)
    lightgbm_tuned_defaults(results)


if __name__ == "__main__":
    main()
