"""
How good is LightGBM with the best published fixed settings? Uses LGBM-TD ("tuned defaults") from
Holzmüller, Grinsztajn & Steinwart, "Better by Default" (NeurIPS 2024), via their pytabkit package.

Those settings were tuned on datasets with at least 2,500 rows, so most of ours are smaller than anything
they saw. pytabkit holds out 20% of the training fold for early stopping and does not refit, as in the paper.
Run from the repo root: uv run --with pytabkit==1.8.0 python -m checks.lightgbm_tuned_defaults
Writes results/2026/lightgbm_tuned_defaults.csv (about 25 minutes for all 142 datasets).
"""

import lightgbm as lgb
from pytabkit import LGBM_TD_Classifier

from utils import N_JOBS, fit_then_predict, run_benchmark


class _QuietLogger:
    """Drop LightGBM's log lines: pytabkit cannot pass verbose=-1, and tiny datasets print a warning per tree."""

    def info(self, msg):
        pass

    def warning(self, msg):
        pass


lgb.register_logger(_QuietLogger())


def fit(X_train, y_train, random_state):
    return LGBM_TD_Classifier(random_state=random_state, n_threads=N_JOBS).fit(X_train, y_train)


if __name__ == "__main__":
    run_benchmark("lightgbm_tuned_defaults", fit_then_predict(fit))
