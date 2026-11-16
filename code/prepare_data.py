"""Build a labeled AnnData file from 10x gene-expression matrices."""

import argparse
from hashlib import file_digest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmread


def read_table(path, required):
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    missing = set(required) - set(table.columns)
    if missing or table.empty:
        raise ValueError(f"{path}: missing columns {sorted(missing)} or empty table")
    if (table[required] == "").any().any():
        raise ValueError(f"{path}: required columns contain empty fields")
    return table


def source_record(path, role, sample_id=""):
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    return {"sample_id": sample_id, "role": role, "path": str(path.resolve()), "sha256": digest}


def read_sample(row, base, labels, args):
    features = pd.read_csv(base / row.features, sep="\t", header=None, dtype=str)
    barcodes = pd.read_csv(base / row.barcodes, sep="\t", header=None, dtype=str)[0]
    if features.shape[1] < 2 or features.iloc[:, :2].isna().any().any():
        raise ValueError(f"{row.sample_id}: expected feature ID and gene symbol columns")
    if barcodes.isna().any() or barcodes.duplicated().any():
        raise ValueError(f"{row.sample_id}: invalid or duplicate barcodes")

    counts = sparse.csr_matrix(mmread(base / row.matrix).T)
    if counts.shape != (len(barcodes), len(features)):
        raise ValueError(f"{row.sample_id}: matrix must be features by barcodes")
    if not np.isfinite(counts.data).all() or (counts.data < 0).any() or (counts.data % 1).any():
        raise ValueError(f"{row.sample_id}: matrix must contain nonnegative integer UMI counts")
    counts.eliminate_zeros()
    if features.shape[1] >= 3:
        is_gene = features[2].eq("Gene Expression").to_numpy()
        counts = counts[:, is_gene]
        features = features.loc[is_gene]
    if features.empty or features[0].duplicated().any():
        raise ValueError(f"{row.sample_id}: gene expression features are missing or IDs are duplicated")

    obs = pd.DataFrame(index=pd.Index(barcodes.to_numpy(), name="barcode"))
    obs["cell_type"] = labels["cell_type"].reindex(obs.index, fill_value="")
    obs["sample_id"] = row.sample_id
    obs["patient_id"] = row.patient_id
    obs["tissue_context"] = row.tissue_context
    obs["barcode"] = obs.index
    obs.index = pd.Index(row.sample_id + ":" + barcodes, name="cell_id")
    obs["total_umis"] = np.asarray(counts.sum(axis=1)).ravel()
    obs["detected_genes"] = counts.getnnz(axis=1)
    mito_genes = features[1].str.upper().str.startswith("MT-").to_numpy()
    mito_umis = np.asarray(counts[:, mito_genes].sum(axis=1)).ravel()
    obs["mito_fraction"] = mito_umis / np.maximum(obs["total_umis"], 1)
    if "qc_exclude" in labels:
        excluded = labels["qc_exclude"].reindex(barcodes, fill_value="0").eq("1").to_numpy()
    else:
        excluded = np.zeros(len(barcodes), dtype=bool)

    quality = ((obs["detected_genes"] >= args.min_genes)
               & (obs["detected_genes"] <= args.max_genes)
               & (obs["total_umis"] >= args.min_umis)
               & (obs["mito_fraction"] <= args.max_mito_fraction)
               & ~excluded)
    labeled = obs["cell_type"].ne("") & obs["cell_type"].ne("unlabeled")
    keep = (quality & labeled).to_numpy()
    if not keep.any():
        raise ValueError(f"{row.sample_id}: no labeled cells after QC; check barcodes and thresholds")
    missing_labels = len(labels.index.difference(barcodes))
    var = pd.DataFrame({"gene_symbol": features[1].to_numpy()}, index=pd.Index(features[0], name="gene_id"))
    dataset = ad.AnnData(X=counts[keep], obs=obs.loc[keep], var=var)
    census = {"sample_id": row.sample_id, "patient_id": row.patient_id,
              "tissue_context": row.tissue_context, "n_raw": len(obs), "n_matched_labels": int(labeled.sum()),
              "n_labels_without_barcode_match": missing_labels, "n_qc_pass": int(quality.sum()),
              "n_retained": int(keep.sum()), "n_excluded_by_annotation": int(excluded.sum())}
    return dataset, census


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=6000)
    parser.add_argument("--min-umis", type=int, default=1)
    parser.add_argument("--max-mito-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if not 1 <= args.min_genes <= args.max_genes or args.min_umis < 1 or not 0 <= args.max_mito_fraction <= 1:
        parser.error("invalid QC thresholds")
    samples = read_table(args.samples, ["sample_id", "patient_id", "tissue_context", "matrix", "features", "barcodes"])
    labels = read_table(args.labels, ["sample_id", "barcode", "cell_type"])
    if samples["sample_id"].duplicated().any() or labels.duplicated(["sample_id", "barcode"]).any():
        parser.error("duplicate sample IDs or duplicate cell label keys")
    if "qc_exclude" in labels and not labels["qc_exclude"].isin(["0", "1"]).all():
        parser.error("qc_exclude must be 0 or 1 when supplied")
    unknown = set(labels["sample_id"]) - set(samples["sample_id"])
    if unknown:
        parser.error(f"labels reference samples absent from the manifest: {sorted(unknown)}")
    label_groups = {sample: part.set_index("barcode") for sample, part in labels.groupby("sample_id")}
    datasets, census = [], []
    sources = [source_record(args.samples, "sample_manifest"), source_record(args.labels, "cell_labels")]
    for row in samples.itertuples(index=False):
        if row.sample_id not in label_groups:
            parser.error(f"{row.sample_id}: no rows in the labels file")
        dataset, record = read_sample(row, args.samples.parent, label_groups[row.sample_id], args)
        datasets.append(dataset)
        census.append(record)
        sources.extend(source_record(args.samples.parent / getattr(row, role), role, row.sample_id)
                       for role in ("matrix", "features", "barcodes"))
    combined = ad.concat(datasets, join="inner", merge="same")
    if not combined.n_vars:
        parser.error("no common gene IDs across samples")
    combined.layers["counts"] = combined.X.copy()
    combined.uns["qc_thresholds"] = {"min_genes": args.min_genes, "max_genes": args.max_genes,
                                      "min_umis": args.min_umis, "max_mito_fraction": args.max_mito_fraction}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    combined.write_h5ad(args.output_dir / "cohort.h5ad", compression="gzip")
    pd.DataFrame(census).to_csv(args.output_dir / "sample_qc.tsv", sep="\t", index=False)
    pd.DataFrame(sources).to_csv(args.output_dir / "source_files.tsv", sep="\t", index=False)
    pd.crosstab(combined.obs["patient_id"], combined.obs["cell_type"]).to_csv(
        args.output_dir / "patient_label_coverage.tsv", sep="\t"
    )
    print(f"{combined.n_obs:,} cells, {combined.n_vars:,} shared genes -> {args.output_dir}")


if __name__ == "__main__":
    main()
