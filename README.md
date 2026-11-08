SmallDatasetBenchmarks
======================
This repo tests machine learning models on small classification datasets. The blog post that summarizes the experiments is here: https://www.data-cowboys.com/blog/which-machine-learning-classifiers-are-best-for-small-datasets/

The original 2020 run compared linear SVM, logistic regression, random forest, LightGBM, AutoGluon, and MLJAR. The 2026 update reruns all six with current libraries and adds four models that need no tuning:

- **LightGBM** with default settings and 1,000 trees (`04_lightgbm_default.py`).
- **CatBoost** with default settings (`05_catboost.py`).
- **TabPFN-3.5**, a pretrained tabular foundation model (`06_tabpfn.py`). Its weights use a non-commercial license: accept it at https://ux.priorlabs.ai and set `TABPFN_TOKEN` before running.
- **TabICLv2**, an open-weight tabular foundation model (`07_tabicl.py`).

AutoGluon 1.6 (`03_autogluon.py`) and MLJAR 1.3 (`08_mljar.py`) get 60 seconds per outer fold. The 2020 AutoGluon and MLJAR runs had 300 seconds, and their results in `results/*.pickle` used the same datasets, folds, and seeds, so `figures.py` includes them for comparison. MLJAR is told to use 10-fold CV, the validation its Compete mode documents: with a time limit, it otherwise falls back to a 90/10 split when its first model trains slowly, which left about 4 validation rows on the smallest datasets. `01_compare_baseline_models.py` is ported to current libraries. Across all 142 datasets, logistic regression and random forest reproduce the 2020 mean AUROC to four decimal places except on `nursery-4class`. SVC differs on 8 datasets, by up to 0.04. The port caps libsvm's iterations, because linear SVC otherwise runs for hours on `autoUniv-au4-2500`; the cap is the likely cause of the SVC differences, but that is unconfirmed.

Setup
-----
Every model uses the same nested cross-validation: 4 stratified outer folds with a fixed seed, plus inner 4-fold CV for models that are tuned. Scores are weighted one-vs-rest AUROC. Datasets larger than 10,000 rows are subsampled to 10,000. The splits are IID, so the conclusions are unlikely to hold when test or production data shifts away from the training data.

Running
-------
Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
uv run python 01_compare_baseline_models.py   # SVC, logistic regression, random forest
uv run python 02_lightgbm.py                  # LightGBM tuned with hyperopt
uv run python 03_autogluon.py                 # AutoGluon, 60 s per fold
uv run python 04_lightgbm_default.py          # LightGBM defaults, 1,000 trees
uv run python 05_catboost.py                  # CatBoost defaults
uv run python 06_tabpfn.py                    # needs a CUDA GPU
uv run python 07_tabicl.py                    # needs a CUDA GPU
uv run python 08_mljar.py                     # MLJAR, 60 s per fold
uv run python 09_timing.py --device cpu       # fit and predict times; then --device gpu
uv run python figures.py                      # tables and plots in results/2026/
```

Each script appends per-fold scores and timings to `results/2026/<model>.csv` and skips datasets already there, so an interrupted run resumes. AutoGluon and MLJAR write many model files: set `AUTOGLUON_TMP_DIR` and `MLJAR_TMP_DIR` to a local folder outside any synced drive. `09_timing.py` times one outer fold of the 30 datasets in `timing_datasets.txt` and writes `results/2026/timing.csv`. `pyproject.toml` pulls CUDA 12.8 builds of PyTorch; change the index if you need a different CUDA version.

Checks
------
The post's checks live in `checks/`. Run them from the repo root as modules, after the benchmark:

```bash
uv run python -m checks.summary                  # numbers for the checks below, from saved results
uv run python -m checks.contamination_check      # memorization: shuffled labels and 10 or 20 training rows
uv run python -m checks.contamination_summary    # summarize that run
uv run python -m checks.autogluon_budget_check   # AutoGluon 60 s vs 300 s on its 10 biggest losses
uv run python -m checks.mljar_split_validation   # the MLJAR benchmark with MLJAR's own validation choice
uv run python -m checks.mljar_budget_check       # that setting at 60 s vs 300 s on two tiny datasets
uv run --with pytabkit==1.8.0 python -m checks.lightgbm_tuned_defaults  # "Better by Default" LightGBM
```

`checks.summary` also compares the rerun baselines with `results/01_compare_baseline_models.pickle`. Every check writes its own CSV in `results/2026/`, and those CSVs are committed, so `checks.summary` and `checks.contamination_summary` reproduce the post's numbers without rerunning anything.

Data
----
The data is a subset of this dataset-of-datasets: "UCI++, a huge collection of preprocessed datasets for supervised classification problems in ARFF format
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.13748.svg)](http://dx.doi.org/10.5281/zenodo.13748)"

`datasets.txt` lists the 142 datasets evaluated. `figures.py` then drops near-duplicate variants, datasets where every model scores 0.99 AUROC or more, and datasets where no model reaches 0.6, as the 2020 post did. These checks use a fixed set of models, so adding a model does not change which datasets count.

Note that UCI++ reuses the same datasets in different configurations and often you can't tell what's a categorical feature. Every feature is treated as continuous after one-hot encoding the nominal ones.
