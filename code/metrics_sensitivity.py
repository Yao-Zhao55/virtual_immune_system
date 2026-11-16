"""Sensitivity of the model comparison to tiny per-patient classes.

Patient-level macro-F1 averages over whatever classes happen to appear in the
held-out patient. P02 has exactly one Epithelial cell, so its F1 can move the
whole patient score by ~0.1. This recomputes the same out-of-fold predictions
with a minimum-support filter and reports overall accuracy too.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
FOLDS = sorted(path.parent.name for path in ROOT.glob("runs/main3_P*/oof_predictions.tsv.gz"))
MODELS = ["logistic", "mlp", "deep_sets", "transformer"]
MIN_SUPPORT = 30


def main() -> int:
    frames = []
    for fold in FOLDS:
        path = ROOT / "runs" / fold / "oof_predictions.tsv.gz"
        frames.append(pd.read_csv(path, sep="\t"))
    data = pd.concat(frames, ignore_index=True)
    print(f"oof predictions: {len(data):,} cells, patients={sorted(data.patient_id.unique())}")

    rows = []
    for (seed, model, patient), part in data.groupby(["seed", "model", "patient_id"]):
        y, p = part["cell_type"].to_numpy(), part["predicted_type"].to_numpy()
        support = pd.Series(y).value_counts()
        keep = set(support[support >= MIN_SUPPORT].index)
        mask = np.array([label in keep for label in y])
        rows.append({
            "seed": seed, "model": model, "patient_id": patient,
            "macro_f1_all": f1_score(y, p, average="macro", labels=np.unique(y), zero_division=0),
            "macro_f1_support30": f1_score(y[mask], p[mask], average="macro",
                                           labels=sorted(keep), zero_division=0),
            "accuracy": float((y == p).mean()),
            "balanced_accuracy": float(np.mean([
                (p[y == label] == label).mean() for label in np.unique(y)
            ])),
            "n_cells": len(y),
            "n_classes_kept": len(keep),
        })
    metrics = pd.DataFrame(rows)

    for column in ("macro_f1_all", "macro_f1_support30", "accuracy", "balanced_accuracy"):
        pivot = metrics.pivot_table(index="patient_id", columns="model", values=column)[MODELS]
        mean = pivot.mean().sort_values(ascending=False)
        print(f"\n== {column}: per patient ==")
        print(pivot.round(4).to_string())
        print("mean:", "  ".join(f"{model}={value:.4f}" for model, value in mean.items()))
        delta = pivot["transformer"] - pivot["mlp"]
        print(f"transformer-mlp: mean={delta.mean():+.4f} per-patient={delta.round(4).to_dict()}")

    print("\n== classes kept per patient (support >= 30) ==")
    print(metrics.pivot_table(index="patient_id", columns="model", values="n_classes_kept").iloc[:, 0].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
