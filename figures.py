"""Summarize the 2026 results (plus the 2020 AutoGluon and MLJAR runs) into tables and plots."""

import os
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils import RESULTS_DIR, dataset_names

MODEL_NAMES = {
    "lightgbm_hyperopt_50": "LightGBM (hyperopt, 50 trials)",
    "lightgbm_default_1000": "LightGBM (default, 1000 trees)",
    "autogluon_1_6_sec_60": "AutoGluon 1.6 (60s)",
    "catboost_default": "CatBoost (default)",
    "tabpfn_v3_5": "TabPFN-3.5",
    "tabicl_v2": "TabICLv2",
    "mljar_1_3_sec_60": "MLJAR 1.3 (60s)",
    # The 2020 baselines, rerun in 2026 with the same script; 2020 recorded no timings.
    "svc": "SVC",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}
# The 2020 runs used the same datasets, folds, and seeds, so they stay comparable.
RESULTS_2020 = {
    "AutoGluon (2020, 300s)": "results/03_autogluon_sec_300.pickle",
    "MLJAR (2020, 300s)": "results/04_mljar_sec_300.pickle",
}
# Pick the datasets with a fixed set of models, so adding a model cannot change which datasets count.
FILTER_MODELS = [
    "SVC",
    "Logistic Regression",
    "Random Forest",
    *RESULTS_2020,
    "LightGBM (hyperopt, 50 trials)",
    "CatBoost (default)",
    "TabPFN-3.5",
    "TabICLv2",
]
# UCI++ has many near-duplicate variants; the 2020 post dropped these.
DUPED_DATASETS = [
    "wine-quality-white-5class",
    "waveform-v2",
    "volcanoes-a3",
    "volcanoes-b3",
    "volcanoes-b4",
    "volcanoes-b5",
    "volcanoes-b6",
    "volcanoes-d3",
    "volcanoes-d4",
    "statlog-german-credit-numeric",
    "thyroid-allhyper",
    "thyroid-allhypo",
    "thyroid-allrep",
    "thyroid-hypothyroid",
    "thyroid-dis",
]


def load_results() -> pd.DataFrame:
    """Return one row per (model, dataset) with mean, std, and total seconds over the outer folds."""
    frames = []
    # Read only the listed models: results/2026 also holds timing.csv and other side runs.
    for key, model in MODEL_NAMES.items():
        path = f"{RESULTS_DIR}/{key}.csv"
        if not os.path.exists(path) or os.path.getsize(path) == 0:  # not started, or just started
            continue
        df = pd.read_csv(path).dropna(subset=["auroc"])
        df["model"] = model
        frames.append(df)
    names = dataset_names()
    scores_2020, seconds_2020 = {}, {}  # seconds per dataset, on 2020 hardware
    for model, path in RESULTS_2020.items():
        with open(path, "rb") as f:
            scores_2020[model], seconds_2020[model] = pickle.load(f)
    for model, scores in scores_2020.items():
        for dataset, folds in zip(names, scores):
            frames.append(pd.DataFrame({"dataset": dataset, "fold": range(len(folds)), "auroc": folds, "model": model}))
    folds = pd.concat(frames)
    summary = folds.groupby(["model", "dataset"]).agg(
        mean_auroc=("auroc", "mean"),
        std_auroc=("auroc", "std"),
        # The 2020 AutoML runs recorded time per dataset, not per fold; they are filled in below.
        seconds=("seconds", lambda s: s.sum(min_count=1)),
        n_folds=("auroc", "size"),
    )
    for model, seconds in seconds_2020.items():
        summary.loc[[(model, dataset) for dataset in names], "seconds"] = seconds
    return summary[summary.n_folds == 4].reset_index()


def filter_datasets(results: pd.DataFrame) -> pd.DataFrame:
    """Keep datasets every model finished, minus duplicates and ones that cannot separate the models.

    Drops datasets where every FILTER_MODELS model is near perfect (all >= 0.99) or where none finds
    signal (best < 0.6).
    """
    n_models = results.model.nunique()
    complete = results.groupby("dataset").model.nunique() == n_models
    results = results[results.dataset.isin(complete[complete].index)]
    results = results[~results.dataset.isin(DUPED_DATASETS)]
    scores = results[results.model.isin(FILTER_MODELS)].groupby("dataset").mean_auroc
    keep = (scores.min() < 0.99) & (scores.max() >= 0.6)
    return results[results.dataset.isin(keep[keep].index)]


def counts_within(results: pd.DataFrame, best: bool) -> Counter:
    """Count datasets where each model is within 0.5% of the best (or worst) mean AUROC."""
    counts = Counter()
    for _, df in results.groupby("dataset"):
        if best:
            counts.update(df.loc[df.mean_auroc >= df.mean_auroc.max() * 0.995, "model"])
        else:
            counts.update(df.loc[df.mean_auroc <= df.mean_auroc.min() * 1.005, "model"])
    return counts


def drop_unfinished(results: pd.DataFrame) -> pd.DataFrame:
    """Drop models still running, so partial runs don't shrink the shared dataset set."""
    n_datasets = len(dataset_names())
    counts = results.groupby("model").dataset.nunique()
    unfinished = counts[counts < n_datasets]
    for model, n in unfinished.items():
        print(f"Skipping {model}: {n}/{n_datasets} datasets done.")
    return results[~results.model.isin(unfinished.index)]


def main() -> None:
    database = pd.read_json("database.json").T
    database["nrow"] = np.minimum(10000, database["nrow"].astype(int))
    results = filter_datasets(drop_unfinished(load_results())).join(database[["nrow"]], on="dataset")
    n_datasets = results.dataset.nunique()
    small = results[results.nrow < 250]

    table = pd.DataFrame(
        {
            "mean AUROC": results.groupby("model").mean_auroc.mean(),
            "median AUROC": results.groupby("model").mean_auroc.median(),
            "mean AUROC (n < 250)": small.groupby("model").mean_auroc.mean(),
            "best or within 0.5%": pd.Series(counts_within(results, best=True)),
            "worst or within 0.5%": pd.Series(counts_within(results, best=False)),
            "median seconds per dataset": results.groupby("model").seconds.median(),
        }
    ).sort_values("mean AUROC", ascending=False)
    table = table.fillna({"best or within 0.5%": 0, "worst or within 0.5%": 0})
    text = f"{n_datasets} datasets ({small.dataset.nunique()} with fewer than 250 rows).\n\n"
    # One format per column, index first: tabulate turns every row into floats.
    text += table.to_markdown(floatfmt=("", ".3f", ".3f", ".3f", ".0f", ".0f", ".1f")) + "\n"
    text += timing_table()
    with open(f"{RESULTS_DIR}/summary.md", "w") as f:
        f.write(text)
    print(text)

    sns.set_theme(context="poster")
    order = table.index.tolist()

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.boxenplot(data=results, x="mean_auroc", y="model", order=order, ax=ax)
    ax.set(xlabel="mean AUROC over outer folds", ylabel="")
    fig.savefig(f"{RESULTS_DIR}/auroc_by_model.png", bbox_inches="tight", dpi=100)

    for y, ylabel, name in [
        ("mean_auroc", "mean AUROC", "auroc_vs_nrow.png"),
        ("std_auroc", "std of AUROC across folds", "std_vs_nrow.png"),
    ]:
        g = sns.lmplot(
            data=results, x="nrow", y=y, hue="model", hue_order=order, lowess=True, height=10, aspect=1.6,
            scatter_kws={"alpha": 0.4},
        )  # fmt: skip
        g.set(xscale="log", xlabel="number of rows", ylabel=ylabel)
        if y == "mean_auroc":
            g.set(ylim=[0.825, 0.95])  # same axis as the 2020 post
        g.savefig(f"{RESULTS_DIR}/{name}", dpi=100)

    plot_timing()


def timing_table() -> str:
    """Return the median train and predict seconds per fold, per device, from 09_timing.py, as markdown."""
    path = f"{RESULTS_DIR}/timing.csv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ""
    timing = pd.read_csv(path)
    timing["predict_ms_per_row"] = 1000 * timing.predict_seconds / timing.n_test
    stats = timing.groupby(["model", "device"])[["fit_seconds", "predict_seconds", "predict_ms_per_row"]].median()
    table = stats.unstack("device").sort_values(("fit_seconds", "cpu"))
    table.columns = [f"{device} median {column}" for column, device in table.columns]
    n = timing.dataset.nunique()
    return f"\nTiming: one outer fold of {n} datasets, medians.\n\n" + table.to_markdown(floatfmt=".4g") + "\n"


def pareto_front(times: pd.Series, auroc: pd.Series) -> list[str]:
    """Return the models no other model beats on both time and AUROC, fastest first."""
    front, best = [], -np.inf
    for model in times.sort_values().index:
        if auroc[model] > best:
            front.append(model)
            best = auroc[model]
    return front


def plot_timing() -> None:
    """Plot AUROC against train and predict time on CPU and GPU, with the Pareto front, from 09_timing.py."""
    path = f"{RESULTS_DIR}/timing.csv"
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    timing = pd.read_csv(path)
    stats = timing.groupby(["device", "model"]).agg(
        auroc=("auroc", "mean"), train=("fit_seconds", "median"), predict=("predict_seconds", "median")
    )
    cpu = stats.loc["cpu"]
    # Models without GPU support run on the CPU of the same machine, so the GPU panels show them at CPU speed.
    gpu = pd.concat([stats.loc["gpu"], cpu.drop(stats.loc["gpu"].index)])
    on_gpu = set(stats.loc["gpu"].index)
    front_color, other_color = "#c8553d", "0.55"
    sns.set_theme(context="poster", style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(24, 20), sharey=True)
    panels = [("cpu", cpu, "24-core CPU"), ("gpu", gpu, "RTX 3080 GPU")]
    for row, (device, df, hardware) in enumerate(panels):
        for col, (column, label) in enumerate([("train", "train"), ("predict", "predict")]):
            ax = axes[row, col]
            front = pareto_front(df[column], df.auroc)
            ax.step(
                df.loc[front, column], df.loc[front, "auroc"], where="post", color=front_color, linewidth=3, zorder=1
            )
            for model, point in df.iterrows():
                color = front_color if model in front else other_color
                hollow = device == "gpu" and model not in on_gpu
                ax.scatter(
                    point[column], point.auroc, s=220, zorder=2, linewidths=3, edgecolors=color,
                    facecolors="white" if hollow else color,
                )  # fmt: skip
                dx, dy = LABEL_OFFSETS.get((device, column, model), (14, 8))
                ax.annotate(
                    model, (point[column], point.auroc), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left", fontsize=17, color="0.15",
                )  # fmt: skip
            ax.set(xscale="log", xlabel=f"median {label} seconds per fold")
            ax.set_title(f"{hardware}: {label}", fontsize=22)
            ax.margins(x=0.35, y=0.08)
        axes[row, 0].set_ylabel("mean AUROC")
    handles = [
        plt.Line2D([], [], color=front_color, linewidth=3, marker="o", markersize=14),
        plt.Line2D([], [], color=other_color, linewidth=0, marker="o", markersize=14),
        plt.Line2D(
            [],
            [],
            color=other_color,
            linewidth=0,
            marker="o",
            markersize=14,
            markerfacecolor="white",
            markeredgewidth=3,
        ),
    ]
    labels = [
        "Pareto front: nothing is both faster and more accurate",
        "beaten on both time and AUROC",
        "no GPU support: runs on the CPU",
    ]
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=20)
    n_train, n_test = timing.n_train.median(), timing.n_test.median()
    title = f"Accuracy vs time: one outer fold of {timing.dataset.nunique()} datasets"
    fig.suptitle(f"{title} (median {n_train:,.0f} train rows, {n_test:,.0f} test rows)", y=0.94)
    fig.savefig(f"{RESULTS_DIR}/timing.png", bbox_inches="tight", dpi=100)


# Move labels off their neighbours, in points; any label not listed sits up and to the right.
LABEL_OFFSETS = {
    ("cpu", "train", "CatBoost (default)"): (14, -32),
    ("gpu", "train", "CatBoost (default)"): (14, -32),
    ("gpu", "train", "LightGBM (hyperopt, 50 trials)"): (-14, 8),
    ("cpu", "train", "Random Forest"): (14, -32),
    ("gpu", "train", "Random Forest"): (14, -32),
    ("cpu", "predict", "CatBoost (default)"): (-30, -6),
    ("gpu", "predict", "CatBoost (default)"): (-30, -6),
    ("cpu", "predict", "Random Forest"): (14, -32),
    ("gpu", "predict", "Random Forest"): (14, -32),
    ("cpu", "predict", "TabICLv2"): (-14, 8),
    ("gpu", "predict", "TabICLv2"): (-14, -6),
    ("cpu", "predict", "AutoGluon 1.6 (60s)"): (-14, 8),
    ("cpu", "train", "Logistic Regression"): (10, -34),
    ("gpu", "train", "Logistic Regression"): (10, -34),
    ("cpu", "predict", "Logistic Regression"): (10, -34),
    ("gpu", "predict", "Logistic Regression"): (10, -34),
}


if __name__ == "__main__":
    main()
