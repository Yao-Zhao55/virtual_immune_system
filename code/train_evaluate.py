"""Compare cell-type classifiers with nested patient-grouped validation."""

import argparse
import contextlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch import nn


MODELS = ("logistic", "mlp", "deep_sets", "transformer")


class GeneSetClassifier(nn.Module):
    def __init__(self, model, n_genes, n_classes):
        super().__init__()
        width = 128
        self.gene_id = nn.Embedding(n_genes, width)
        self.expression = nn.Linear(1, width)
        self.register_buffer("gene_indices", torch.arange(n_genes), persistent=False)
        if model == "transformer":
            self.blocks = nn.ModuleList([
                nn.TransformerEncoderLayer(width, 4, dim_feedforward=256, dropout=0.1,
                                           activation="gelu", batch_first=True, norm_first=True)
                for _ in range(4)
            ])
        else:
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 512), nn.GELU(),
                              nn.Dropout(0.1), nn.Linear(512, width), nn.Dropout(0.1))
                for _ in range(4)
            ])
        self.use_attention = model == "transformer"
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, n_classes))

    def forward(self, values, gene_indices=None):
        indices = self.gene_indices if gene_indices is None else gene_indices
        tokens = self.gene_id(indices) + self.expression(values.unsqueeze(-1))
        for block in self.blocks:
            updated = block(tokens)
            tokens = updated if self.use_attention else tokens + updated
        return self.head(tokens.mean(dim=1))


def choose_features(matrix, var_names, train, evaluation, n_genes):
    training = ad.AnnData(matrix[train], var=pd.DataFrame(index=var_names))
    sc.pp.highly_variable_genes(training, flavor="seurat", n_top_genes=n_genes)
    ranked = training.var["dispersions_norm"].to_numpy()
    eligible = np.flatnonzero(np.isfinite(ranked) & np.isfinite(training.var["dispersions"].to_numpy()))
    if len(eligible) < n_genes:
        raise ValueError(f"only {len(eligible)} eligible training genes; lower --n-genes")
    order = np.argsort(-ranked[eligible], kind="stable")[:n_genes]
    columns = eligible[order]
    return matrix[train][:, columns].toarray(), matrix[evaluation][:, columns].toarray(), var_names[columns].tolist()


def grouped_scores(y, predicted, groups):
    scores = {}
    for patient in np.unique(groups):
        mask = groups == patient
        scores[str(patient)] = float(f1_score(y[mask], predicted[mask], labels=np.unique(y[mask]),
                                              average="macro", zero_division=0))
    return scores


def amp_context(device, amp):
    """Opt-in bf16 autocast; the default keeps the pre-registered fp32 path."""
    if amp == "bf16" and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def predict_torch(model, values, batch_size, device, amp="off"):
    model.eval()
    with torch.inference_mode(), amp_context(device, amp):
        return np.concatenate([
            model(batch.to(device)).float().softmax(dim=1).cpu().numpy()
            for batch in torch.from_numpy(values).split(batch_size)
        ])


def fit_model(name, parameter, x_train, y_train, x_eval, y_eval, groups_eval, args, seed, epochs=None):
    start = perf_counter()
    if name == "logistic":
        model = LogisticRegression(C=parameter, max_iter=1000, random_state=seed)
        model.fit(x_train, y_train)
        fit_seconds = perf_counter() - start
        start = perf_counter()
        probabilities = model.predict_proba(x_eval)
        state = {"coef": model.coef_, "intercept": model.intercept_, "classes": model.classes_}
        return probabilities, state, 0, fit_seconds, perf_counter() - start, None

    torch.manual_seed(seed)
    device = args.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    n_genes = x_train.shape[1]
    if name == "mlp":
        model = nn.Sequential(nn.Linear(n_genes, 512), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, args.n_classes))
    else:
        model = GeneSetClassifier(name, n_genes, args.n_classes)
    model.to(device)
    base_model = model
    if getattr(args, "compile", False) and name in ("deep_sets", "transformer"):
        try:
            model = torch.compile(model, dynamic=False)
            print(f"torch.compile enabled for {name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - compile is an optimisation only
            print(f"torch.compile unavailable ({type(exc).__name__}: {exc}); using eager", flush=True)
            model = base_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameter, weight_decay=0.01)
    train_values, train_labels = torch.from_numpy(x_train), torch.from_numpy(y_train)
    loss_function = nn.CrossEntropyLoss()
    best_score, best_epoch = -np.inf, 0
    generator = torch.Generator().manual_seed(seed)
    max_epochs = args.max_epochs if epochs is None else epochs
    for epoch in range(1, max_epochs + 1):
        model.train()
        for row_ids in torch.randperm(len(train_values), generator=generator).split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, args.amp):
                loss = loss_function(model(train_values[row_ids].to(device)), train_labels[row_ids].to(device))
            loss.backward()
            optimizer.step()
        if epochs is not None:
            continue
        probabilities = predict_torch(model, x_eval, args.batch_size, device, args.amp)
        score = np.mean(list(grouped_scores(y_eval, probabilities.argmax(axis=1), groups_eval).values()))
        if score > best_score:
            best_score, best_epoch, best_probabilities = score, epoch, probabilities
        if epoch - best_epoch >= args.patience:
            break
    if epochs is None:
        return best_probabilities, None, best_epoch, None, None, None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    probabilities = predict_torch(model, x_eval, args.batch_size, device, args.amp)
    inference_seconds = perf_counter() - start
    if isinstance(model, GeneSetClassifier):
        with torch.inference_mode():
            values = torch.from_numpy(x_eval[:8]).to(device)
            order = torch.arange(n_genes - 1, -1, -1, device=device)
            torch.testing.assert_close(base_model(values), base_model(values[:, order], order), atol=2e-5, rtol=2e-5)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    source = getattr(model, "_orig_mod", model)
    state = {key: value.cpu() for key, value in source.state_dict().items()}
    return probabilities, state, epochs, fit_seconds, inference_seconds, peak


def inner_splits(y, groups, outer_train, n_splits, split_seed):
    labels = np.unique(y)
    for offset in range(20):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=split_seed + offset)
        splits = [(outer_train[train], outer_train[validation])
                  for train, validation in splitter.split(outer_train, y[outer_train], groups[outer_train])]
        if all(np.array_equal(np.unique(y[train]), labels) for train, _ in splits):
            return splits
    raise ValueError("inner training folds lack one or more classes; review patient_label_coverage.tsv")


def run_outer(matrix, data, y, groups, patient, splits, args, seed, fold_dir):
    train, test = np.flatnonzero(groups != patient), np.flatnonzero(groups == patient)
    grids = {name: (0.1, 1.0) if name == "logistic" else (1e-4, 3e-4) for name in MODELS}
    tuning = {(name, parameter): {"scores": {}, "epochs": []}
              for name, grid in grids.items() for parameter in grid}
    for inner_train, inner_val in splits:
        x_train, x_val, _ = choose_features(matrix, data.var_names, inner_train, inner_val, args.n_genes)
        for (name, parameter), result in tuning.items():
            probs, _, epoch, *_ = fit_model(name, parameter, x_train, y[inner_train], x_val, y[inner_val],
                                            groups[inner_val], args, seed)
            result["scores"].update(grouped_scores(y[inner_val], probs.argmax(axis=1), groups[inner_val]))
            result["epochs"].append(epoch)
    x_train, x_test, genes = choose_features(matrix, data.var_names, train, test, args.n_genes)
    fold_dir.mkdir(parents=True)
    pd.Series(genes, name="gene_id").to_csv(fold_dir / "genes.tsv", sep="\t", index=False)
    config = []
    predictions = []
    metadata = data.obs.iloc[test][["patient_id", "sample_id", "tissue_context", "cell_type"]].copy()
    metadata.insert(0, "cell_id", metadata.index)
    for name, grid in grids.items():
        parameter = max(grid, key=lambda value: np.mean(list(tuning[name, value]["scores"].values())))
        epochs = int(round(np.median(tuning[name, parameter]["epochs"])))
        probs, state, _, fit_seconds, infer_seconds, peak = fit_model(
            name, parameter, x_train, y[train], x_test, y[test], groups[test], args, seed, epochs
        )
        if name == "logistic":
            np.savez(fold_dir / "logistic.npz", **state)
        else:
            torch.save(state, fold_dir / f"{name}.pt")
        rows = metadata.assign(model=name, seed=seed, predicted_type=args.classes[probs.argmax(axis=1)])
        for col, label in enumerate(args.classes):
            rows[f"p:{label}"] = probs[:, col]
        predictions.append(rows)
        config.append({"model": name, "parameter_name": "C" if name == "logistic" else "learning_rate",
                       "parameter": parameter, "epochs": epochs, "fit_seconds": fit_seconds,
                       "inference_seconds": infer_seconds, "cells_per_second": len(test) / max(infer_seconds, 1e-12),
                       "peak_cuda_allocated_bytes": peak,
                       "inner_candidates": [{"parameter": value, **tuning[name, value]} for value in grid]})
    write_json(fold_dir / "models.json", config)
    return pd.concat(predictions, ignore_index=True)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_reports(predictions, args):
    patient_rows, class_rows, delta_rows, composition = [], [], [], []
    predicted_col = "predicted_type"
    for (seed, model), rows in predictions.groupby(["seed", "model"]):
        y, p, g = (rows[col].to_numpy() for col in ("cell_type", predicted_col, "patient_id"))
        patient_rows.extend({"seed": seed, "model": model, "patient_id": patient, "macro_f1": score}
                            for patient, score in grouped_scores(y, p, g).items())
        precision, recall, f1, support = precision_recall_fscore_support(y, p, labels=args.classes, zero_division=0)
        for i, label in enumerate(args.classes):
            class_rows.append({"seed": seed, "model": model, "cell_type": label, "precision": precision[i],
                               "recall": recall[i], "f1": f1[i], "support": support[i],
                               "patient_support": len(np.unique(g[y == label]))})
        pd.DataFrame(confusion_matrix(y, p, labels=args.classes), index=args.classes, columns=args.classes).to_csv(
            args.output_dir / f"confusion_{seed}_{model}.tsv", sep="\t", index_label="reference"
        )
        for (patient, sample, tissue), subset in rows.groupby(["patient_id", "sample_id", "tissue_context"], observed=True):
            reference = subset["cell_type"].value_counts(normalize=True)
            predicted = subset[predicted_col].value_counts(normalize=True)
            composition.extend({"seed": seed, "model": model, "patient_id": patient, "sample_id": sample,
                                "tissue_context": tissue, "cell_type": label, "n_cells": len(subset),
                                "reference_fraction": reference.get(label, 0.0),
                                "predicted_fraction": predicted.get(label, 0.0)} for label in args.classes)
    patient_scores = pd.DataFrame(patient_rows)
    for seed, rows in patient_scores.groupby("seed"):
        pivot = rows.pivot(index="patient_id", columns="model", values="macro_f1")
        for baseline in ("mlp", "deep_sets", "logistic"):
            differences = (pivot["transformer"] - pivot[baseline]).to_numpy()
            rng = np.random.default_rng(args.split_seed)
            bootstrap = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(axis=1)
            low, high = np.quantile(bootstrap, [0.025, 0.975]) if len(differences) > 1 else (np.nan, np.nan)
            delta_rows.append({"seed": seed, "baseline": baseline, "n_patients": len(differences),
                               "delta_macro_f1": differences.mean(), "improved_patients": (differences > 0).sum(),
                               "paired_patient_ci_low": low, "paired_patient_ci_high": high})
    reports = {"patient_scores": patient_scores, "class_scores": pd.DataFrame(class_rows),
               "model_deltas": pd.DataFrame(delta_rows), "sample_composition": pd.DataFrame(composition),
               "model_scores": patient_scores.groupby(["seed", "model"], as_index=False)["macro_f1"].mean(),
               "paired_tissue_composition": paired_composition(pd.DataFrame(composition))}
    for name, table in reports.items():
        table.to_csv(args.output_dir / f"{name}.tsv", sep="\t", index=False)


def paired_composition(composition):
    keys = ["seed", "model", "patient_id", "cell_type"]
    contexts = ["adjacent", "primary_tumor"]
    columns = keys + ["reference_delta", "predicted_delta", "delta_error"]
    subset = composition[composition["tissue_context"].isin(contexts)].copy()
    if subset.empty:
        return pd.DataFrame(columns=columns)
    fractions = ["reference_fraction", "predicted_fraction"]
    subset[fractions] = subset[fractions].mul(subset["n_cells"], axis=0)
    totals = subset.groupby(keys + ["tissue_context"], observed=True)[fractions + ["n_cells"]].sum()
    paired = totals[fractions].div(totals["n_cells"], axis=0).unstack("tissue_context")
    paired = paired.reindex(columns=pd.MultiIndex.from_product([fractions, contexts])).dropna()
    reference_delta = paired["reference_fraction"]["primary_tumor"] - paired["reference_fraction"]["adjacent"]
    predicted_delta = paired["predicted_fraction"]["primary_tumor"] - paired["predicted_fraction"]["adjacent"]
    return pd.DataFrame({"reference_delta": reference_delta, "predicted_delta": predicted_delta,
                         "delta_error": predicted_delta - reference_delta}).reset_index()[columns]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Labeled cohort.h5ad from prepare_data.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-genes", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the gene-set models; identical maths, faster steps")
    parser.add_argument("--amp", choices=["off", "bf16"], default="off",
                        help="bf16 autocast for neural models; off keeps the pre-registered fp32 path")
    parser.add_argument("--pilot-patient", help="Run one outer fold for development only")
    args = parser.parse_args()
    if min(args.n_genes, args.max_epochs, args.patience, args.batch_size) < 1 or args.inner_folds < 2:
        parser.error("invalid model, batch or fold size")
    if len(set(args.seeds)) != len(args.seeds) or min(*args.seeds, args.split_seed) < 0:
        parser.error("seeds must be nonnegative and training seeds must be unique")
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but PyTorch cannot access it")
    args.device = torch.device(device)
    data = ad.read_h5ad(args.input)
    label_col = "cell_type"
    required_cols = {"patient_id", "sample_id", "tissue_context", label_col, "total_umis"}
    if not required_cols.issubset(data.obs) or "counts" not in data.layers:
        parser.error("input must be the prepared cohort.h5ad")
    args.classes, y = np.unique(data.obs[label_col].astype(str).to_numpy(), return_inverse=True)
    y = y.astype(np.int64)
    args.n_classes = len(args.classes)
    groups = data.obs["patient_id"].astype(str).to_numpy()
    patients = np.unique(groups)
    if len(patients) <= args.inner_folds or args.n_classes < 2 or any(np.unique(groups[y == i]).size < 3 for i in range(args.n_classes)):
        parser.error("need enough patients for nested folds and each cell type must occur in at least three patients")
    if args.pilot_patient and args.pilot_patient not in patients:
        parser.error(f"unknown pilot patient: {args.pilot_patient}")
    held_out = [args.pilot_patient] if args.pilot_patient else patients.tolist()
    schedules, split_rows = {}, []
    for fold, patient in enumerate(held_out, start=1):
        train = np.flatnonzero(groups != patient)
        splits = inner_splits(y, groups, train, args.inner_folds, args.split_seed)
        schedules[patient] = splits
        split_rows.extend({"outer_fold": fold, "held_out_patient": patient, "inner_fold": inner,
                           "patient_id": group, "role": role}
                          for inner, (inner_train, inner_val) in enumerate(splits, start=1)
                          for role, members in (("train", groups[inner_train]), ("validation", groups[inner_val]),
                                                ("test", [patient])) for group in np.unique(members))
    totals = data.obs["total_umis"].to_numpy(dtype=np.float32)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0):
        parser.error("every retained cell must have a positive finite total UMI count")
    counts = sparse.csr_matrix(data.layers["counts"], dtype=np.float32)
    matrix = counts.multiply((10000 / totals)[:, None]).tocsr()
    np.log1p(matrix.data, out=matrix.data)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(split_rows).to_csv(args.output_dir / "patient_splits.tsv", sep="\t", index=False)
    write_json(args.output_dir / "run_config.json", {
        "arguments": {key: str(value) if isinstance(value, (Path, torch.device)) else value
                      for key, value in vars(args).items() if key != "classes"},
        "classes": args.classes.tolist(), "cuda_device": torch.cuda.get_device_name() if device == "cuda" else None,
        "versions": {package: version(package) for package in ("numpy", "pandas", "scipy", "scikit-learn", "scanpy", "anndata", "torch")},
    })
    predictions = []
    for seed in args.seeds:
        for fold, patient in enumerate(held_out, start=1):
            print(f"seed={seed} outer fold {fold}/{len(held_out)} held-out patient={patient}", flush=True)
            fold_dir = args.output_dir / f"seed_{seed}" / f"fold_{fold:02d}"
            fold_predictions = run_outer(matrix, data, y, groups, patient, schedules[patient], args, seed, fold_dir)
            fold_predictions.to_csv(fold_dir / "predictions.tsv.gz", sep="\t", index=False)
            predictions.append(fold_predictions)
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(args.output_dir / "oof_predictions.tsv.gz", sep="\t", index=False)
    write_reports(predictions, args)


if __name__ == "__main__":
    main()
