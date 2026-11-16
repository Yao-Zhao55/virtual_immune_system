"""Time the expensive parts of train_evaluate on the real cohort.

Answers two questions before committing to a multi-hour nested CV run:
  * how long does one linear-baseline fit take on a full training fold,
  * how long does one epoch of the transformer cost on this GPU.
"""

import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
COHORT = ROOT / "runs" / "prepared" / "cohort.h5ad"
N_GENES = 512


def main() -> int:
    t0 = time.perf_counter()
    data = ad.read_h5ad(COHORT)
    print(f"load h5ad: {time.perf_counter() - t0:.1f}s  {data.n_obs:,} x {data.n_vars:,}", flush=True)

    y = pd.Categorical(data.obs["cell_type"].astype(str)).codes.astype(np.int64)
    patients = data.obs["patient_id"].astype(str).to_numpy()
    train = np.flatnonzero(patients != "P01")
    print(f"train cells: {len(train):,}", flush=True)

    from scipy import sparse

    totals = data.obs["total_umis"].to_numpy(dtype=np.float32)
    counts = sparse.csr_matrix(data.layers["counts"], dtype=np.float32)
    matrix = counts.multiply((10000 / totals)[:, None]).tocsr()
    np.log1p(matrix.data, out=matrix.data)

    t0 = time.perf_counter()
    training = ad.AnnData(matrix[train], var=pd.DataFrame(index=data.var_names))
    sc.pp.highly_variable_genes(training, flavor="seurat", n_top_genes=N_GENES)
    columns = np.argsort(-training.var["dispersions_norm"].to_numpy(), kind="stable")[:N_GENES]
    x = matrix[train][:, columns].toarray()
    print(f"feature selection + densify: {time.perf_counter() - t0:.1f}s  {x.shape}", flush=True)

    for name, c in (("logistic C=0.1", 0.1), ("logistic C=1.0", 1.0)):
        t0 = time.perf_counter()
        LogisticRegression(C=c, max_iter=1000, random_state=17).fit(x, y[train])
        print(f"{name}: {time.perf_counter() - t0:.1f}s", flush=True)

    device = torch.device("cuda")
    for batch in (128, 256):
        for name in ("mlp", "transformer"):
            if name == "mlp":
                model = nn.Sequential(nn.Linear(N_GENES, 512), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(256, y.max() + 1)).to(device)
                forward = lambda values: model(values)  # noqa: E731
            else:
                width = 128
                gene_id = nn.Embedding(N_GENES, width).to(device)
                expression = nn.Linear(1, width).to(device)
                blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(width, 4, dim_forward := 256, dropout=0.1,
                                               activation="gelu", batch_first=True, norm_first=True).to(device)
                    for _ in range(4)
                ])
                head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, y.max() + 1)).to(device)
                indices = torch.arange(N_GENES, device=device)

                def forward(values, gene_id=gene_id, expression=expression, blocks=blocks, head=head, indices=indices):
                    tokens = gene_id(indices) + expression(values.unsqueeze(-1))
                    for block in blocks:
                        tokens = block(tokens)
                    return head(tokens.mean(dim=1))

            optimizer = torch.optim.AdamW(list(model.parameters()) if name == "mlp" else
                                          (list(gene_id.parameters()) + list(expression.parameters())
                                           + list(blocks.parameters()) + list(head.parameters())),
                                          lr=3e-4, weight_decay=0.01)
            loss_fn = nn.CrossEntropyLoss()
            tensors = torch.from_numpy(x[:40000])
            labels = torch.from_numpy(y[train][:40000])
            t0 = time.perf_counter()
            model.train()
            for ids in torch.randperm(len(tensors)).split(batch):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(forward(tensors[ids].to(device)), labels[ids].to(device))
                loss.backward()
                optimizer.step()
            seconds = time.perf_counter() - t0
            steps = len(tensors) // batch
            print(f"{name} batch={batch}: {seconds:.1f}s for {steps} steps on 40k cells "
                  f"-> {seconds / 40000 * len(train):.1f}s/epoch on {len(train):,} cells", flush=True)
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
