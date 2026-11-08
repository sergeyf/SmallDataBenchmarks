"""
How good is CatBoost with no tuning at all?
"""

from catboost import CatBoostClassifier

from utils import N_JOBS, fit_then_predict, run_benchmark


def fit(X_train, y_train, random_state, device="cpu"):
    model = CatBoostClassifier(
        random_seed=random_state,
        verbose=0,
        thread_count=N_JOBS,
        allow_writing_files=False,
        task_type="GPU" if device == "gpu" else "CPU",
    )
    return model.fit(X_train, y_train)


if __name__ == "__main__":
    run_benchmark("catboost_default", fit_then_predict(fit))
