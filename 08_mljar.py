"""
How does MLJAR 1.x compare? Same Compete mode as 2020, with 60 seconds per outer fold to match AutoGluon 1.6,
and 10-fold CV inside it.
"""

import os
import shutil
import tempfile

import pandas as pd
from supervised import AutoML

from utils import N_JOBS, fit_then_predict, run_benchmark

SEC = 60
# Model files churn heavily: point this at a local, unsynced folder on the same drive as the repo (Windows).
TMP_DIR = os.environ.get("MLJAR_TMP_DIR") or None
# Compete documents 10-fold CV, but with a time limit it keeps a 90/10 split unless its first decision tree
# trains in under SEC/250 s. Here that tree takes 1-7 s, so the smallest datasets picked models on about
# 4 validation rows and lost up to 0.14 AUROC. "auto" restores MLJAR's own choice, for checks/.
VALIDATION = {"validation_type": "kfold", "k_folds": 10, "shuffle": True, "stratify": True}


class FittedMljar:
    """A fitted AutoML that returns class-ordered probabilities and deletes its model files afterwards."""

    def __init__(self, automl, path, columns):
        self.automl, self.path, self.columns = automl, path, columns

    def predict_proba(self, X_test):
        try:
            predictions = self.automl.predict_all(pd.DataFrame(X_test, columns=self.columns))
            # Columns are "prediction_<label>"; order them 0..k-1 to match the other models.
            labels = sorted(int(c.removeprefix("prediction_")) for c in predictions if c.startswith("prediction_"))
            return predictions[[f"prediction_{label}" for label in labels]].to_numpy()
        finally:
            shutil.rmtree(self.path, ignore_errors=True)


def fit(X_train, y_train, random_state):
    columns = [f"f{i}" for i in range(X_train.shape[1])]
    binary = len(set(y_train)) == 2
    if TMP_DIR:
        os.makedirs(TMP_DIR, exist_ok=True)
    path = tempfile.mkdtemp(prefix="mljar_", dir=TMP_DIR)
    automl = AutoML(
        results_path=os.path.join(path, "run"),  # MLJAR wants a folder that does not exist yet
        mode="Compete",
        ml_task="binary_classification" if binary else "multiclass_classification",
        # MLJAR still has no multiclass AUROC metric, so multiclass optimizes log loss, as in 2020.
        eval_metric="auc" if binary else "logloss",
        validation_strategy=VALIDATION,
        total_time_limit=SEC,
        n_jobs=N_JOBS,
        random_state=random_state,
        verbose=0,
    )
    automl.fit(pd.DataFrame(X_train, columns=columns), pd.Series(y_train))
    return FittedMljar(automl, path, columns)


if __name__ == "__main__":
    run_benchmark(f"mljar_1_3_sec_{SEC}", fit_then_predict(fit))
