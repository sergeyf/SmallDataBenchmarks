"""Is AutoGluon 1.6's drop from 2020 on its 10 worst datasets due to 60 s instead of 300 s?

Reruns 03_autogluon.py's fit at 60 s (a control for run-to-run noise, since the 10 were picked as worst)
and at 300 s, on all 4 outer folds, and keeps each fold's leaderboard before the files are deleted.
Run from the repo root on an otherwise idle machine:
AUTOGLUON_TMP_DIR=<local folder> uv run python -m checks.autogluon_budget_check
Writes results/2026/autogluon_budget_check.csv (scores) and autogluon_budget_leaderboards.csv.
"""

import importlib.util
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils import RESULTS_DIR, load_data, subsample, weighted_auroc

# The 10 largest drops from AutoGluon (2020, 300s) to AutoGluon 1.6 (60s) on the 106 filtered datasets.
DATASETS = [
    "hill-valley-without-noise",
    "hill-valley-with-noise",
    "wine-quality-5class",
    "bupa",
    "appendicitis",
    "habermans-survival",
    "thoracic-surgery",
    "arcene",
    "connectionist-mines-vs-rocks",
    "autoUniv-au6-cd1-400",
]
BUDGETS = [60, 300]


def main() -> None:
    spec = importlib.util.spec_from_file_location("autogluon", "03_autogluon.py")
    autogluon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(autogluon)
    scores, boards = [], []
    for budget in BUDGETS:
        autogluon.SEC = budget  # fit() reads the module-level budget
        for name in DATASETS:
            X, y = subsample(*load_data(name))
            for fold, (train, test) in enumerate(StratifiedKFold(n_splits=4, shuffle=True, random_state=0).split(X, y)):
                start = time.perf_counter()
                fitted = autogluon.fit(X[train], y[train], 0)
                board = fitted.predictor.leaderboard()  # read before predict_proba deletes the model files
                auroc = weighted_auroc(y[test], fitted.predict_proba(X[test]))
                seconds = time.perf_counter() - start
                scores.append({"dataset": name, "budget": budget, "fold": fold, "auroc": auroc, "seconds": seconds})
                boards.append(board.assign(dataset=name, budget=budget, fold=fold))
                print(
                    f"{name} {budget}s fold {fold}: auroc={auroc:.3f}, {len(board)} models, {seconds:.0f}s", flush=True
                )
                pd.DataFrame(scores).to_csv(f"{RESULTS_DIR}/autogluon_budget_check.csv", index=False)
                pd.concat(boards).to_csv(f"{RESULTS_DIR}/autogluon_budget_leaderboards.csv", index=False)


if __name__ == "__main__":
    main()
