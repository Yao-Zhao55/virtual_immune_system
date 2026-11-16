"""Benchmark the transformer block in fp32 vs bf16 at 512 genes.

Decides whether the full protocol is affordable on an 8 GB laptop GPU, and
whether a larger batch fits in bf16 (attention cost grows with batch x 512^2).
"""

import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
COHORT = ROOT / "runs" / "prepared" / "cohort.h5ad"
N_GENES = 512
WIDTH, LAYERS = 128, 4


def build_matrix():
    data = ad.read_h5ad(COHORT)
    totals = data.obs["total_umis"].to_numpy(dtype=np.float32)
    counts = sparse.csr_matrix(data.layers["counts"], dtype=np.float32)
    matrix = counts.multiply((10000 / totals)[:, None]).tocsr()
    np.log1p(matrix.data, out=matrix.data)
    y = pd.Categorical(data.obs["cell_type"].astype(str)).codes.astype(np.int64)
    train = np.flatnonzero(data.obs["patient_id"].astype(str).to_numpy() != "P01")
    training = ad.AnnData(matrix[train], var=pd.DataFrame(index=data.var_names))
    sc.pp.highly_variable_genes(training, flavor="seurat", n_top_genes=N_GENES)
    columns = np.argsort(-training.var["dispersions_norm"].to_numpy(), kind="stable")[:N_GENES]
    x = matrix[train][:, columns].toarray()
    return x, y[train], len(train)


def make_model(n_classes):
    gene_id = nn.Embedding(N_GENES, WIDTH).cuda()
    expression = nn.Linear(1, WIDTH).cuda()
    blocks = nn.ModuleList([nn.TransformerEncoderLayer(WIDTH, 4, dim_feedforward=256, dropout=0.1,
                                                       activation="gelu", batch_first=True,
                                                       norm_first=True).cuda() for _ in range(LAYERS)])
    head = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, n_classes)).cuda()
    indices = torch.arange(N_GENES, device="cuda")

    def forward(values):
        tokens = gene_id(indices) + expression(values.unsqueeze(-1))
        for block in blocks:
            tokens = block(tokens)
        return head(tokens.mean(dim=1))

    params = list(gene_id.parameters()) + list(expression.parameters()) + list(blocks.parameters()) + list(head.parameters())
    return forward, params


def main() -> int:
    x, y, n_train = build_matrix()
    n_classes = int(y.max()) + 1
    sample = 40000
    tensors = torch.from_numpy(x[:sample])
    labels = torch.from_numpy(y[:sample])
    print(f"matrix ready: {x.shape}, extrapolating to {n_train:,} training cells", flush=True)

    for amp in ("fp32", "bf16"):
        for batch in (256, 512):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            forward, params = make_model(n_classes)
            optimizer = torch.optim.AdamW(params, lr=3e-4, weight_decay=0.01)
            loss_fn = nn.CrossEntropyLoss()
            try:
                t0 = time.perf_counter()
                for ids in torch.randperm(len(tensors)).split(batch):
                    optimizer.zero_grad(set_to_none=True)
                    if amp == "bf16":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            loss = loss_fn(forward(tensors[ids].cuda()), labels[ids].cuda())
                    else:
                        loss = loss_fn(forward(tensors[ids].cuda()), labels[ids].cuda())
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                seconds = time.perf_counter() - t0
                per_epoch = seconds / sample * n_train
                peak = torch.cuda.max_memory_allocated() / 2**30
                print(f"{amp:5s} batch={batch:3d}: {seconds:5.1f}s/{sample // batch} steps  "
                      f"-> {per_epoch:6.1f}s per epoch on {n_train:,} cells  peak={peak:.2f} GB", flush=True)
            except torch.OutOfMemoryError:
                print(f"{amp:5s} batch={batch:3d}: OOM", flush=True)
            del forward, params, optimizer
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
