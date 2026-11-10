#!/usr/bin/env python
"""
Standard memory-conscious Scanpy workflow for a large scRNA-seq dataset
(~673k cells x ~22.5k genes) on a 32 GB RAM machine.

Steps: load -> QC metrics -> filtering -> normalize/log1p -> HVG selection (flags only)
       -> raw = all-gene log-normalized (built without a full extra copy)
       -> subset to HVGs -> scale -> PCA -> neighbors -> UMAP -> Leiden -> save
"""

import os
import gc
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import anndata as ad

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
INPUT_PATH = "/scratch/home/lfung/CS223_class/CS223_project/data/broad1.h5ad"
OUTPUT_DIR = "/scratch/home/lfung/CS223_class/CS223_project/output"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "processed_dataset.h5ad")

MIN_GENES = 200          # min genes per cell
MIN_CELLS = 3            # min cells per gene
MAX_PCT_MT = 10.0        # max % mitochondrial counts
TARGET_SUM = 1e4         # normalization target
N_TOP_HVG = 2000         # number of highly variable genes
MAX_SCALE_VALUE = 10     # clip value after scaling
N_PCS = 30               # PCs used for the neighbor graph
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
RANDOM_STATE = 0

sc.settings.verbosity = 2

# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
print("Loading data...")
adata = sc.read_h5ad(INPUT_PATH)
print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

# Defensive: unique names (needed for gene-based operations)
if not adata.var_names.is_unique:
    print("Warning: var_names not unique; making unique.")
    adata.var_names_make_unique()
if not adata.obs_names.is_unique:
    adata.obs_names_make_unique()

# Defensive: keep sparse CSR format (efficient for row-wise ops); do not densify.
if not sp.issparse(adata.X):
    print("X is dense; converting to CSR sparse to save memory.")
    adata.X = sp.csr_matrix(adata.X)
elif not sp.isspmatrix_csr(adata.X) and adata.X.format != "csr":
    print(f"X is {adata.X.format}; converting to CSR.")
    adata.X = adata.X.tocsr()

# Use float32 to limit memory (only if not already float32)
if adata.X.dtype != np.float32:
    adata.X = adata.X.astype(np.float32)

# Defensive: check that X looks like raw counts (non-negative integers)
sample = adata.X.data[: min(100000, adata.X.data.size)]
if sample.size > 0:
    if sample.min() < 0:
        print("WARNING: negative values found in X; data may already be scaled.")
    elif not np.allclose(sample, np.round(sample)):
        print("WARNING: X contains non-integer values; data may already be "
              "normalized. Proceeding with normalization anyway.")
del sample

# ----------------------------------------------------------------------------
# 2. QC metrics (mitochondrial, ribosomal, hemoglobin)
# ----------------------------------------------------------------------------
gene_names_upper = adata.var_names.str.upper()

# Mitochondrial genes: 'MT-' (human) or 'mt-' (mouse); case-insensitive check
adata.var["mt"] = gene_names_upper.str.startswith("MT-")
# Ribosomal genes: RPS / RPL
adata.var["ribo"] = gene_names_upper.str.match(r"^RP[SL]\d")
# Hemoglobin genes: HB* excluding HBP (e.g. HBP1)
adata.var["hb"] = gene_names_upper.str.match(r"^HB[^P]")

n_mt, n_ribo, n_hb = (int(adata.var[c].sum()) for c in ("mt", "ribo", "hb"))
print(f"Gene groups found -> mito: {n_mt}, ribo: {n_ribo}, hb: {n_hb}")

qc_vars = [c for c in ("mt", "ribo", "hb") if adata.var[c].any()]
if n_mt == 0:
    print("WARNING: No mitochondrial genes (MT-/mt-) found; "
          "mitochondrial filtering will be skipped.")

# percent_top=None avoids extra expensive computations
sc.pp.calculate_qc_metrics(
    adata, qc_vars=qc_vars, percent_top=None, log1p=False, inplace=True
)

# ----------------------------------------------------------------------------
# 3. Filtering
# ----------------------------------------------------------------------------
n0 = adata.n_obs
sc.pp.filter_cells(adata, min_genes=MIN_GENES)          # in place
print(f"Removed {n0 - adata.n_obs} cells with < {MIN_GENES} genes")

if "pct_counts_mt" in adata.obs.columns:
    keep = (adata.obs["pct_counts_mt"] < MAX_PCT_MT).values
    print(f"Removing {int((~keep).sum())} cells with >= {MAX_PCT_MT}% mito counts")
    # In-place-ish subsetting: a single copy of the filtered matrix, then free the old one
    adata = adata[keep].copy()
    del keep
    gc.collect()

g0 = adata.n_vars
sc.pp.filter_genes(adata, min_cells=MIN_CELLS)          # in place
print(f"Removed {g0 - adata.n_vars} genes expressed in < {MIN_CELLS} cells")
print(f"After filtering: {adata.n_obs} cells x {adata.n_vars} genes")

if adata.n_obs == 0 or adata.n_vars == 0:
    raise RuntimeError("No cells or genes left after filtering; check thresholds.")

# ----------------------------------------------------------------------------
# 4. Normalization + log transform (in place on sparse X; no counts layer kept)
# ----------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
sc.pp.log1p(adata)

# ----------------------------------------------------------------------------
# 5. Highly variable genes
#    subset=False -> only flags are written to adata.var; no matrix copy.
# ----------------------------------------------------------------------------
sc.pp.highly_variable_genes(
    adata, n_top_genes=N_TOP_HVG, flavor="seurat", subset=False
)
hvg_mask = adata.var["highly_variable"].values.copy()
print(f"Selected {int(hvg_mask.sum())} highly variable genes")

# ----------------------------------------------------------------------------
# 6. Build adata.raw (all genes, log-normalized, UNSCALED) and subset to HVGs.
#    A lightweight AnnData sharing the same X (no copy) is used to store raw;
#    then the working object is reduced to HVGs, so the full matrix is not
#    duplicated more than once. Raw must be set before scaling.
# ----------------------------------------------------------------------------
raw_adata = ad.AnnData(X=adata.X, obs=adata.obs[[]], var=adata.var.copy())
adata_hvg = adata[:, hvg_mask].copy()      # only HVG columns are copied
adata_hvg.raw = raw_adata                  # all-gene log-normalized expression
del raw_adata, adata
gc.collect()
adata = adata_hvg
del adata_hvg
gc.collect()
print(f"Working object: {adata.n_obs} x {adata.n_vars}; "
      f"raw: {adata.raw.n_vars} genes")

# ----------------------------------------------------------------------------
# 7. Scale (zero mean, unit variance, clip at 10). Densifies the HVG matrix only
#    (~673k x 2000 float32 ≈ 5.4 GB), which is unavoidable for zero-centering.
# ----------------------------------------------------------------------------
sc.pp.scale(adata, zero_center=True, max_value=MAX_SCALE_VALUE)

# ----------------------------------------------------------------------------
# 8. PCA, neighbors, UMAP
# ----------------------------------------------------------------------------
n_pcs = min(N_PCS, adata.n_vars - 1, adata.n_obs - 1)
sc.tl.pca(adata, n_comps=max(n_pcs, 50) if adata.n_vars > 50 else n_pcs,
          svd_solver="arpack", random_state=RANDOM_STATE)

sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=n_pcs,
                random_state=RANDOM_STATE)
sc.tl.umap(adata, random_state=RANDOM_STATE)

# ----------------------------------------------------------------------------
# 9. Leiden clustering (igraph flavor: faster and lighter for large data)
# ----------------------------------------------------------------------------
sc.tl.leiden(
    adata,
    resolution=LEIDEN_RESOLUTION,
    key_added="leiden",
    flavor="igraph",
    n_iterations=2,
    directed=False,
    random_state=RANDOM_STATE,
)
print(f"Leiden found {adata.obs['leiden'].nunique()} clusters")

# ----------------------------------------------------------------------------
# 10. Save
# ----------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
adata.write_h5ad(OUTPUT_PATH)
print(f"Saved processed dataset to: {OUTPUT_PATH}")
