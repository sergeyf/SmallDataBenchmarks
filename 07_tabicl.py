"""
How good is TabICLv2, an open-weight tabular foundation model, with no tuning?
"""

from tabicl import TabICLClassifier

from utils import fit_then_predict, run_benchmark


def fit(X_train, y_train, random_state, device="gpu"):
    model = TabICLClassifier(device="cuda" if device == "gpu" else "cpu", random_state=random_state)
    return model.fit(X_train, y_train)


if __name__ == "__main__":
    run_benchmark("tabicl_v2", fit_then_predict(fit))
