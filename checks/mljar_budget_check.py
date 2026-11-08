"""Did MLJAR 1.3's split-validation run lose to MLJAR 2020 on tiny datasets for lack of time?

Runs 08_mljar.py's fit with MLJAR's own validation choice (a 90/10 split here) at 60 s and 300 s, on all
4 outer folds of two tiny datasets. Keeps MLJAR's leaderboard (models fitted, validation scores, training
times) before the model files are deleted: its decision-tree train_time shows why MLJAR kept the split.
Run from the repo root on an otherwise idle machine:
MLJAR_TMP_DIR=<local folder> uv run python -m checks.mljar_budget_check
Writes results/2026/mljar_budget_check.csv (scores) and mljar_budget_leaderboards.csv.
"""

import importlib.util
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import RESULTS_DIR, load_data, subsample, weighted_auroc

DATASETS = ["glioma16", "robot-failure-lp1"]
BUDGETS = [60, 300]


def main() -> None:
    spec = importlib.util.spec_from_file_location("mljar", "08_mljar.py")
    mljar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mljar)
    mljar.VALIDATION = "auto"  # the setting the first 60 s run had
    scores, boards = [], []
    for name in DATASETS:
        X, y = subsample(*load_data(name))
        for budget in BUDGETS:
            mljar.SEC = budget  # fit() reads the module-level budget
            for fold, (train, test) in enumerate(StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(X, y)):
                start = time.perf_counter()
                fitted = mljar.fit(X[train], y[train], 0)
                board = fitted.automl.get_leaderboard()  # read before predict_proba deletes the model files
                auroc = weighted_auroc(y[test], fitted.predict_proba(X[test]))
                seconds = time.perf_counter() - start
                scores.append({"dataset": name, "budget": budget, "fold": fold, "auroc": auroc, "seconds": seconds})
                boards.append(board.assign(dataset=name, budget=budget, fold=fold))
                print(
                    f"{name} {budget}s fold {fold}: auroc={auroc:.3f}, {len(board)} models, {seconds:.0f}s",
                    flush=True,
                )
                pd.DataFrame(scores).to_csv(f"{RESULTS_DIR}/mljar_budget_check.csv", index=False)
                pd.concat(boards).to_csv(f"{RESULTS_DIR}/mljar_budget_leaderboards.csv", index=False)


if __name__ == "__main__":
    main()
