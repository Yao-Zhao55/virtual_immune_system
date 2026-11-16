"""Derive a reproducible reference cell-type annotation for GSE184362.

The GEO series ships raw 10x matrices only: there is no per-cell label table,
and the authors' repository contains marker code but no annotation file. This
script therefore builds an *explicitly reference-derived* annotation:

    QC -> normalize -> HVG -> PCA -> neighbours -> Leiden -> marker scoring

Every cluster assignment lands in data/annotation_clusters.tsv together with
the per-cluster marker scores, so labels can be audited and re-derived. This is
a reference annotation, not pathologist ground truth: downstream results
measure reproduction of *this* annotation, exactly as the plan states.

Usage:
  python build_annotation.py                      # cluster + annotate
  python build_annotation.py --reuse-cache        # reuse saved clusters
"""

import argparse
import gc
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.io import mmread

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "data" / "samples.tsv"
OUT = ROOT / "data" / "labels.tsv"
AUDIT = ROOT / "data" / "annotation_clusters.tsv"
SCORES = ROOT / "data" / "annotation_marker_scores.tsv"
CACHE = ROOT / "data" / "_clusters.tsv.gz"

# Coarse, marker-based classes. Deliberately conservative: DC/monocyte and
# T/NK splits are deferred until within-cluster marker separation is verified.
MARKERS = {
    "T_NK": ["CD3D", "CD3E", "CD2", "TRAC", "CD247", "IL7R", "CD8A", "NKG7", "GNLY", "KLRD1"],
    "B": ["CD79A", "CD79B", "MS4A1", "CD19", "IGHM", "IGKC", "BANK1"],
    "Plasma": ["MZB1", "JCHAIN", "SDC1", "XBP1", "DERL3"],
    "Myeloid": ["LYZ", "CD68", "CD14", "FCGR3A", "S100A8", "S100A9", "ITGAX", "C1QA", "CSF1R"],
    "Mast": ["TPSAB1", "TPSB2", "CPA3", "MS4A2", "KIT"],
    "Endothelial": ["PECAM1", "VWF", "CLDN5", "EGFL7"],
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB"],
    "Epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "TG", "TPO", "PAX8"],
}


def read_sample(row, base: Path) -> ad.AnnData:
    features = pd.read_csv(base / row.features, sep="\t", header=None, dtype=str)
    barcodes = pd.read_csv(base / row.barcodes, sep="\t", header=None, dtype=str)[0]
    counts = sparse.csr_matrix(mmread(base / row.matrix).T)
    if features.shape[1] >= 3:
        mask = features[2].eq("Gene Expression").to_numpy()
        counts, features = counts[:, mask], features.loc[mask]
    obs = pd.DataFrame(index=pd.Index(row.sample_id + ":" + barcodes, name="cell_id"))
    obs["barcode"] = barcodes.to_numpy()
    obs["sample_id"] = row.sample_id
    obs["patient_id"] = row.patient_id
    obs["tissue_context"] = row.tissue_context
    obs["total_umis"] = np.asarray(counts.sum(axis=1)).ravel()
    obs["detected_genes"] = counts.getnnz(axis=1)
    mito = features[1].str.upper().str.startswith("MT-").to_numpy()
    obs["mito_fraction"] = np.asarray(counts[:, mito].sum(axis=1)).ravel() / np.maximum(obs["total_umis"], 1)
    var = pd.DataFrame({"gene_symbol": features[1].to_numpy()}, index=pd.Index(features[0], name="gene_id"))
    duplicate = var.index.duplicated()
    if duplicate.any():
        counts = counts[:, ~duplicate]
        var = var[~duplicate]
    return ad.AnnData(X=counts, obs=obs, var=var)


def read_cohort(samples: pd.DataFrame, base: Path) -> ad.AnnData:
    parts = []
    for row in samples.itertuples(index=False):
        part = read_sample(row, base)
        print(f"read {row.sample_id} {row.patient_id} {row.tissue_context}: "
              f"{part.n_obs:,} cells x {part.n_vars:,} genes", flush=True)
        parts.append(part)
    data = ad.concat(parts, join="inner", merge="same")
    print(f"concatenated: {data.n_obs:,} cells x {data.n_vars:,} shared genes", flush=True)
    return data


def cluster_cells(work: ad.AnnData, resolution: float, seed: int, cache: Path) -> pd.Series:
    sc.pp.highly_variable_genes(work, flavor="seurat", n_top_genes=2000, batch_key="sample_id")
    hv = work[:, work.var["highly_variable"].to_numpy()].copy()
    dense_bytes = hv.n_obs * hv.n_vars * 4
    if dense_bytes < 2.5e9:
        sc.pp.scale(hv, max_value=10)
        print(f"scaled: dense {dense_bytes / 1e9:.2f} GB", flush=True)
    else:
        print(f"skip scaling: dense would be {dense_bytes / 1e9:.2f} GB", flush=True)
    sc.tl.pca(hv, n_comps=50, svd_solver="arpack", random_state=seed)
    sc.pp.neighbors(hv, n_neighbors=15, n_pcs=40, random_state=seed)
    try:
        sc.tl.leiden(hv, resolution=resolution, key_added="cluster", random_state=seed,
                     flavor="igraph", n_iterations=2, directed=False)
        print("clustering: leiden", flush=True)
    except Exception as exc:  # noqa: BLE001 - leiden needs igraph/leidenalg
        from sklearn.cluster import KMeans

        print(f"leiden unavailable ({type(exc).__name__}: {exc}); k-means fallback", flush=True)
        labels = KMeans(n_clusters=20, random_state=seed, n_init=10).fit_predict(hv.obsm["X_pca"][:, :30])
        hv.obs["cluster"] = pd.Categorical(labels.astype(str))
    clusters = hv.obs["cluster"].astype(str)
    clusters.rename("cluster").to_frame().to_csv(cache, sep="\t")
    print(f"{clusters.nunique()} clusters -> {cache}", flush=True)
    del hv
    gc.collect()
    return clusters


def score_markers(data: ad.AnnData) -> pd.DataFrame:
    """Score each marker module per cluster, using gene IDs as scanpy expects."""
    by_symbol: dict[str, list[str]] = {}
    for gene_id, symbol in zip(data.var_names, data.var["gene_symbol"].to_numpy(), strict=True):
        by_symbol.setdefault(symbol, []).append(gene_id)
    frame = {}
    for name, genes in MARKERS.items():
        ids = sorted({gene_id for gene in genes for gene_id in by_symbol.get(gene, [])})
        missing = [gene for gene in genes if gene not in by_symbol]
        if not ids:
            print(f"marker module {name}: absent from cohort, skipped", flush=True)
            continue
        if missing:
            print(f"marker module {name}: missing {missing}", flush=True)
        sc.tl.score_genes(data, ids, score_name=f"score_{name}", random_state=0)
        frame[name] = data.obs.groupby("cluster", observed=True)[f"score_{name}"].mean()
    return pd.DataFrame(frame)


def assign_labels(scores: pd.DataFrame, min_margin: float) -> dict[str, str]:
    labels = {}
    for cluster, row in scores.iterrows():
        ordered = row.sort_values(ascending=False)
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else -np.inf
        labels[cluster] = ordered.index[0] if best > 0 and best - second >= min_margin else "unlabeled"
    return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--min-margin", type=float, default=0.15,
                        help="required lead of the best marker module over the runner-up")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--reuse-cache", action="store_true", help="reuse clusters from a previous run")
    args = parser.parse_args()

    samples = pd.read_csv(SAMPLES, sep="\t", dtype=str)
    data = read_cohort(samples, SAMPLES.parent)
    qc_pass = ((data.obs["detected_genes"] >= 200) & (data.obs["detected_genes"] <= 6000)
               & (data.obs["total_umis"] >= 1) & (data.obs["mito_fraction"] <= 0.20)).to_numpy()
    print(f"QC pass: {qc_pass.sum():,}/{data.n_obs:,}", flush=True)

    work = data[qc_pass].copy()
    del data
    gc.collect()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)

    if args.reuse_cache and CACHE.exists():
        cached = pd.read_csv(CACHE, sep="\t", index_col=0)["cluster"].astype(str)
        missing = cached.index.difference(work.obs_names)
        if len(missing):
            print(f"cache does not cover {len(missing):,} cells; reclustering", flush=True)
            clusters = cluster_cells(work, args.resolution, args.seed, CACHE)
        else:
            clusters = cached.reindex(work.obs_names)
            print(f"reused {clusters.nunique()} clusters from {CACHE}", flush=True)
    else:
        clusters = cluster_cells(work, args.resolution, args.seed, CACHE)
    work.obs["cluster"] = pd.Categorical(clusters.reindex(work.obs_names).to_numpy())

    scores = score_markers(work)
    scores.to_csv(SCORES, sep="\t")
    labels = assign_labels(scores, args.min_margin)
    print("cluster -> cell_type:",
          {key: labels[key] for key in sorted(labels, key=lambda value: int(value) if value.isdigit() else 0)})

    audit = scores.copy()
    audit.insert(0, "n_cells", work.obs["cluster"].value_counts().sort_index())
    audit.insert(1, "assigned", pd.Series(labels))
    audit.to_csv(AUDIT, sep="\t", index_label="cluster")

    table = pd.DataFrame({
        "sample_id": work.obs["sample_id"].to_numpy(),
        "barcode": work.obs["barcode"].to_numpy(),
        "cell_type": work.obs["cluster"].map(labels).to_numpy(),
        "qc_exclude": "0",
    })
    table.to_csv(OUT, sep="\t", index=False)
    print(f"\nlabels -> {OUT}")
    print(table["cell_type"].value_counts().to_string())
    print("\ncells per sample x class:\n" + pd.crosstab(table["sample_id"], table["cell_type"]).to_string())
    per_patient = pd.crosstab(work.obs["patient_id"].to_numpy(), table["cell_type"].to_numpy())
    print("\npatients per class:\n" + per_patient.gt(0).sum().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
