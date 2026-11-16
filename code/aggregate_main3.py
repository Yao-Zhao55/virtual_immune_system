"""Aggregate the three completed outer folds into one comparison table."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
FOLDS = sorted(path.parent.name for path in ROOT.glob("runs/main3_P*/patient_scores.tsv"))
MODELS = ["logistic", "mlp", "deep_sets", "transformer"]

patient_rows, class_rows, timings = [], [], []
for fold in FOLDS:
    base = ROOT / "runs" / fold
    if not (base / "patient_scores.tsv").exists():
        print(f"missing results: {fold}")
        continue
    scores = pd.read_csv(base / "patient_scores.tsv", sep="\t")
    patient_rows.append(scores)
    class_rows.append(pd.read_csv(base / "class_scores.tsv", sep="\t"))
    fold_dir = next(base.glob("seed_*/fold_*"))
    for config in json.loads((fold_dir / "models.json").read_text(encoding="utf-8")):
        timings.append({
            "fold": fold.split("_")[-1],
            "model": config["model"],
            "epochs": config["epochs"],
            "fit_seconds": config["fit_seconds"],
            "peak_gb": (config["peak_cuda_allocated_bytes"] or 0) / 2**30,
        })

patients = pd.concat(patient_rows, ignore_index=True)
classes = pd.concat(class_rows, ignore_index=True)
timing = pd.DataFrame(timings)

pivot = patients.pivot_table(index="patient_id", columns="model", values="macro_f1")
print("== macro-F1 per held-out patient ==")
print(pivot[MODELS].round(4).to_string())
print("\n== mean over the 3 held-out patients ==")
print(pivot[MODELS].mean().round(4).sort_values(ascending=False).to_string())

print("\n== transformer minus baseline, per patient ==")
for baseline in ("mlp", "deep_sets", "logistic"):
    delta = pivot["transformer"] - pivot[baseline]
    print(f"{baseline:<11} mean={delta.mean():+.4f}  per-patient={delta.round(4).to_dict()}")

print("\n== per-class recall (mean over folds) ==")
recall = classes.pivot_table(index="cell_type", columns="model", values="recall")
print(recall[MODELS].round(3).to_string())

print("\n== per-class F1 (mean over folds) ==")
f1 = classes.pivot_table(index="cell_type", columns="model", values="f1")
print(f1[MODELS].round(3).to_string())

print("\n== fit cost per fold (median across folds) ==")
print(timing.groupby("model")[["epochs", "fit_seconds", "peak_gb"]].median().round(2).to_string())
print(f"\ntotal wall time from log: P01 2h47m, P02 1h22m, P03 1h16m")
