"""Rerun the MLJAR benchmark with MLJAR's own validation choice instead of the forced 10-fold CV.

With a 60 s budget, Compete mode keeps a 90/10 split unless its first decision tree trains in under 0.24 s,
so on this machine every dataset used the split. Compare with summary.py to see what 10-fold changed.
Run from the repo root: MLJAR_TMP_DIR=<local folder> uv run python -m checks.mljar_split_validation
Writes results/2026/mljar_1_3_sec_60_split_validation.csv. Takes about as long as 08_mljar.py (a day).
"""

import importlib.util

from utils import fit_then_predict, run_benchmark

spec = importlib.util.spec_from_file_location("mljar", "08_mljar.py")
mljar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mljar)
mljar.VALIDATION = "auto"  # MLJAR's default; fit() reads the module-level setting

if __name__ == "__main__":
    run_benchmark(f"mljar_1_3_sec_{mljar.SEC}_split_validation", fit_then_predict(mljar.fit))
