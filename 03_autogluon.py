"""
How does AutoGluon 1.x compare? Same best_quality preset as 2020, with 60 seconds per outer fold.
"""

import os
import shutil
import tempfile

import pandas as pd
from autogluon.tabular import TabularPredictor

from utils import N_JOBS, fit_then_predict, run_benchmark

SEC = 60
# Model files churn heavily: point this at a local, unsynced folder on the same drive as the repo (Windows).
TMP_DIR = os.environ.get("AUTOGLUON_TMP_DIR") or None


class FittedAutoGluon:
    """A fitted predictor that returns class-ordered probabilities and deletes its model files afterwards."""

    def __init__(self, predictor, path, columns):
        self.predictor, self.path, self.columns = predictor, path, columns

    def predict_proba(self, X_test):
        try:
            proba = self.predictor.predict_proba(pd.DataFrame(X_test, columns=self.columns))
            # Columns are the class labels; order them 0..k-1 to match the other models.
            return proba[sorted(proba.columns)].to_numpy()
        finally:
            shutil.rmtree(self.path, ignore_errors=True)


def fit(X_train, y_train, random_state, device="cpu"):
    columns = [f"f{i}" for i in range(X_train.shape[1])]
    train = pd.DataFrame(X_train, columns=columns).assign(y=y_train)
    binary = len(set(y_train)) == 2
    if TMP_DIR:
        os.makedirs(TMP_DIR, exist_ok=True)
    path = tempfile.mkdtemp(prefix="autogluon_", dir=TMP_DIR)
    predictor = TabularPredictor(
        label="y",
        problem_type="binary" if binary else "multiclass",
        # In 2020 multiclass used f1_weighted because AutoGluon had no multiclass AUROC; it does now.
        eval_metric="roc_auc" if binary else "roc_auc_ovr_weighted",
        path=path,
        verbosity=0,
        # Dynamic stacking (new since 2020) spends part of a short budget on a trial fit and can run out of time.
    ).fit(
        train,
        time_limit=SEC,
        presets="best_quality",
        dynamic_stacking=False,
        num_cpus=N_JOBS,
        num_gpus=1 if device == "gpu" else 0,
    )
    return FittedAutoGluon(predictor, path, columns)


if __name__ == "__main__":
    run_benchmark(f"autogluon_1_6_sec_{SEC}", fit_then_predict(fit))
