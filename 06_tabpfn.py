"""
How good is TabPFN-3.5, a pretrained tabular foundation model, with no tuning?

The 3.5 weights use a non-commercial license. Accept it at https://ux.priorlabs.ai and set TABPFN_TOKEN
(or log in once interactively) before running.
"""

import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from utils import fit_then_predict, run_benchmark

# GPU settings for a 10 GB card, measured on the RTX 3080 used for the benchmark.
# Predict at most this many test rows per call. Each test row is scored on its own, so results do not change.
PREDICT_CHUNK = 1000
GPU_MEMORY_FRACTION = 0.85  # leaves room for the desktop and the CUDA context
MEMORY_SAVING_ROWS = 1000


class ChunkedTabPFNClassifier(TabPFNClassifier):
    """TabPFN that predicts in chunks, so a 10 GB GPU does not overflow.

    On Windows the NVIDIA driver spills overflowing GPU memory into system RAM instead of failing. On a
    7,500-row training set, a 2,500-row predict then ran about 20x slower than in chunks of 1,000.
    """

    def predict_proba(self, X):
        chunks = [X[i : i + PREDICT_CHUNK] for i in range(0, len(X), PREDICT_CHUNK)]
        return np.concatenate([super().predict_proba(chunk) for chunk in chunks])


def fit(X_train, y_train, random_state, device="gpu"):
    if device == "cpu":
        # Plenty of RAM, and each extra predict call repeats the work over the training rows.
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V3_5, device="cpu", random_state=random_state)
    else:
        # Past the cap PyTorch frees its cache instead of growing into system RAM, which ran up to 6x slower.
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
        model = ChunkedTabPFNClassifier.create_default_for_version(
            ModelVersion.V3_5,
            device="cuda",
            random_state=random_state,
            # The "auto" heuristic is scaled from an 80 GB GPU and leaves this off on a 10 GB card until it
            # overflows. Memory saving costs about 0.5 s on small datasets and nothing measurable on large ones.
            memory_saving_mode=len(X_train) > MEMORY_SAVING_ROWS,
        )
    return model.fit(X_train, y_train)


if __name__ == "__main__":
    run_benchmark("tabpfn_v3_5", fit_then_predict(fit))
