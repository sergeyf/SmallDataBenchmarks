"""
Which baseline models are best? Linear SVM, logistic regression, and random forest with a small grid search.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from utils import N_JOBS, fit_then_predict, run_benchmark

C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]


def grid_search(estimator, param_grid):
    """Build a fit function that tunes on inner 4-fold CV, as in the 2020 run."""

    def fit(X_train, y_train, random_state):
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
        clf = GridSearchCV(estimator, param_grid, cv=inner_cv, scoring="roc_auc_ovr_weighted", n_jobs=N_JOBS)
        return clf.fit(X_train, y_train)

    return fit


svc = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        # Cap the solver: at C >= 10, libsvm runs for hours on autoUniv-au4-2500 without converging.
        # The 2020 run had no cap; with it, the 2020 scores on autoUniv-au4-2500 and meta-data reproduce.
        (
            "svc",
            SVC(
                kernel="linear",
                class_weight="balanced",
                probability=True,
                tol=1e-4,
                random_state=0,
                max_iter=1_000_000,
            ),
        ),
    ]
)
# Without the elastic-net penalty, logistic regression sometimes fails badly.
logistic = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("logistic", LogisticRegression(solver="saga", max_iter=10000, l1_ratio=0.1, random_state=0)),
    ]
)
random_forest = RandomForestClassifier(random_state=0)

FITS = {
    "svc": grid_search(svc, {"svc__C": C_GRID}),
    "logistic_regression": grid_search(logistic, {"logistic__C": C_GRID}),
    "random_forest": grid_search(random_forest, {"max_depth": [1, 2, 4, 8, 16, 32, None]}),
}

if __name__ == "__main__":
    for name, fit in FITS.items():
        run_benchmark(name, fit_then_predict(fit))
