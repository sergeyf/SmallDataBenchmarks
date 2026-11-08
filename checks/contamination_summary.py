"""Summarize results/2026/contamination.csv from contamination_check.py.

Run from the repo root: uv run python -m checks.contamination_summary
"""

import pandas as pd

r = pd.read_csv("results/2026/contamination.csv")
# Average the seeds first, so every dataset counts once.
per = r.groupby(["model", "dataset", "test"]).auroc.mean().unstack("test")
models = ["TabPFN-3.5", "TabICLv2", "CatBoost (default)", "Random Forest (default)"]
print(r.groupby(["model", "test"]).dataset.nunique().unstack("test"), "\n")

shuffled = per.shuffled.unstack("model")[models]
print("Test 1, shuffled training labels: AUROC on the true test labels (0.5 = learned nothing)")
summary = pd.DataFrame(
    {
        "mean": shuffled.mean(),
        "median": shuffled.median(),
        "max": shuffled.max(),
        "datasets > 0.6": (shuffled > 0.6).sum(),
        "datasets > 0.7": (shuffled > 0.7).sum(),
    }
)
print(summary.round(3).to_string(), "\n")
# Paired against the controls on the same shuffles: memorization would put the pretrained models above them.
control = shuffled[["CatBoost (default)", "Random Forest (default)"]].mean(axis=1)
for model in models[:2]:
    gap = shuffled[model] - control
    print(f"{model} minus mean control: mean {gap.mean():+.3f}, datasets with gap > 0.1: {(gap > 0.1).sum()}")
top = shuffled.assign(control=control).sort_values("TabPFN-3.5", ascending=False).head(8)
print("\nHighest TabPFN-3.5 shuffled scores, with the controls on the same shuffles:")
print(top.round(3).to_string(), "\n")

print("Test 2, few training rows: mean AUROC on datasets that have every k (at most 5 classes)")
few = per.dropna(subset=["k=10", "k=20"])
table = few.groupby("model")[["k=10", "k=20", "full"]].mean().loc[models]
# Share of the full-data lift over 0.5 that each model keeps with k rows.
for k in ["k=10", "k=20"]:
    table[f"kept {k}"] = ((few[k] - 0.5) / (few.full - 0.5)).groupby("model").median().loc[models]
print(f"{few.reset_index().dataset.nunique()} datasets")
print(table.round(3).to_string())
