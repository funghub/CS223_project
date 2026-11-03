import os
import gc
import numpy as np
import scipy.sparse as sp
import scanpy as sc

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
INPUT_PATH = "C:/Users/jimmy/Documents/GitHub/CS223_project/data/broad1.h5ad"
OUTPUT_DIR = "C:/Users/jimmy/Documents/GitHub/CS223_project/output/"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "processed_dataset.h5ad")

MIN_GENES = 200
MIN_CELLS = 3
MAX_PCT_MT = 10.0          # keep cells with < 10% mitochondrial counts
TARGET_SUM = 1e4
N_TOP_HVGS = 2000
N_PCS = 30
N_NEIGHBORS = 15
LEIDEN_RESOLUTION = 0.5
RANDOM_STATE = 0

sc.settings.verbosity = 3
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 1. Load data (in memory; backed mode is not needed for filtering/normalizing)
# ----------------------------------------------------------------------------
adata = sc.read_h5ad(INPUT_PATH)
print(f"Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

# Make sure gene names are unique (cheap, no matrix copy)
adata.var_names_make_unique()

# Keep the existing sparse format; only convert dense to CSR if necessary
if not sp.issparse(adata.X):
    print("X is dense; converting to CSR sparse.")
    adata.X = sp.csr_matrix(adata.X)

# Raw counts are expected to be integers; warn if the data looks already normalized
_sample = adata.X[:1000].data if sp.issparse(adata.X) else np.asarray(adata.X[:1000]).ravel()
if _sample.size > 0 and not np.allclose(_sample, np.round(_sample)):
    print("WARNING: X does not appear to contain raw integer counts. "
          "Normalization will still be applied as requested.")
del _sample

# ----------------------------------------------------------------------------
# 2. Basic filtering (genes first, then cells) - in place, no copies
# ----------------------------------------------------------------------------
sc.pp.filter_cells(adata, min_genes=MIN_GENES)
sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
print(f"After min_genes/min_cells filter: {adata.n_obs} cells x {adata.n_vars} genes")

# ----------------------------------------------------------------------------
# 3. QC metrics: mitochondrial, ribosomal, hemoglobin (with defensive checks)
# ----------------------------------------------------------------------------
gene_names = adata.var_names.astype(str)

# Mitochondrial: 'MT-' (human) or 'mt-' (mouse)
adata.var["mt"] = gene_names.str.startswith(("MT-", "mt-"))
# Ribosomal: RPS/RPL (human) or Rps/Rpl (mouse)
adata.var["ribo"] = gene_names.str.startswith(("RPS", "RPL", "Rps", "Rpl"))
# Hemoglobin: HBA/HBB/HBD/HBG... excluding HBP (e.g. HBP1) ; mouse Hba/Hbb
adata.var["hb"] = gene_names.str.match(r"^(HB[ABDEGMQZ]\d*$|Hb[a-z]-?[a-z]*\d*$)") & \
                  ~gene_names.str.startswith(("HBP", "Hbp"))

qc_vars = []
for key, label in [("mt", "mitochondrial"), ("ribo", "ribosomal"), ("hb", "hemoglobin")]:
    n = int(adata.var[key].sum())
    print(f"Found {n} {label} genes")
    if n > 0:
        qc_vars.append(key)

sc.pp.calculate_qc_metrics(adata, qc_vars=qc_vars, percent_top=None,
                           log1p=False, inplace=True)

# ----------------------------------------------------------------------------
# 4. Filter cells on mitochondrial content (only if MT genes exist)
# ----------------------------------------------------------------------------
if "pct_counts_mt" in adata.obs:
    keep = (adata.obs["pct_counts_mt"] < MAX_PCT_MT).values
    print(f"Removing {int((~keep).sum())} cells with >= {MAX_PCT_MT}% mito counts")
    adata = adata[keep].copy()   # one copy to materialize the subset; old one freed below
    gc.collect()
else:
    print("No mitochondrial genes found; skipping mito filter.")

# Second gene filter in case cell removal left rare genes
sc.pp.filter_genes(adata, min_cells=MIN_CELLS)
print(f"After QC filtering: {adata.n_obs} cells x {adata.n_vars} genes")

# ----------------------------------------------------------------------------
# 5. Normalize to 10,000 counts per cell and log1p (in place)
#    No counts layer is stored, to save memory.
# ----------------------------------------------------------------------------
sc.pp.normalize_total(adata, target_sum=TARGET_SUM)
sc.pp.log1p(adata)

# ----------------------------------------------------------------------------
# 6. Highly variable genes (computed on log-normalized data, no subsetting yet)
#    seurat flavor works on log data and needs no extra dependencies.
# ----------------------------------------------------------------------------
sc.pp.highly_variable_genes(adata, n_top_genes=N_TOP_HVGS, flavor="seurat",
                            subset=False, inplace=True)
print(f"Selected {int(adata.var['highly_variable'].sum())} HVGs")

# ----------------------------------------------------------------------------
# 7. Store all-gene log-normalized data in .raw AFTER subsetting to HVGs?
#    Requirement: raw must keep unscaled all-gene expression for DE/markers.
#    To avoid copying the full matrix twice, we build raw from the current
#    full-gene object (a single copy of X), then immediately reduce adata to HVGs
#    so the original full matrix is released.
# ----------------------------------------------------------------------------
hvg_mask = adata.var["highly_variable"].values.copy()

# Build a lightweight AnnData holding the full log-normalized matrix (shares X, no copy)
import anndata as ad
raw_adata = ad.AnnData(X=adata.X, obs=adata.obs[[]], var=adata.var[["gene_ids"]]
                       if "gene_ids" in adata.var else adata.var[[]])
# Subset main object to HVGs (creates the HVG-only matrix)
adata = adata[:, hvg_mask].copy()
adata.raw = raw_adata          # full-gene unscaled log-normalized expression
del raw_adata
gc.collect()
print(f"Main object reduced to HVGs: {adata.n_obs} x {adata.n_vars}; raw has {adata.raw.n_vars} genes")

# ----------------------------------------------------------------------------
# 8. Scale to zero mean / unit variance, clip at 10
#    Scaling makes the matrix dense: 673k x 2000 float32 ~ 5.4 GB (acceptable).
# ----------------------------------------------------------------------------
if sp.issparse(adata.X) and adata.X.dtype != np.float32:
    adata.X = adata.X.astype(np.float32)
sc.pp.scale(adata, max_value=10)
if adata.X.dtype != np.float32:
    adata.X = adata.X.astype(np.float32)

# ----------------------------------------------------------------------------
# 9. PCA, neighbors, UMAP
# ----------------------------------------------------------------------------
sc.tl.pca(adata, n_comps=N_PCS, svd_solver="arpack", random_state=RANDOM_STATE)
sc.pp.neighbors(adata, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS, random_state=RANDOM_STATE)
sc.tl.umap(adata, random_state=RANDOM_STATE)

# ----------------------------------------------------------------------------
# 10. Leiden clustering (resolution 0.5)
# ----------------------------------------------------------------------------
try:
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, random_state=RANDOM_STATE,
                 key_added="leiden", flavor="igraph", n_iterations=2, directed=False)
except TypeError:
    # Older scanpy versions without the flavor argument
    sc.tl.leiden(adata, resolution=LEIDEN_RESOLUTION, random_state=RANDOM_STATE,
                 key_added="leiden")
print(adata.obs["leiden"].value_counts())

# ----------------------------------------------------------------------------
# 11. Save the processed object
# ----------------------------------------------------------------------------
adata.write_h5ad(OUTPUT_PATH, compression="gzip")
print(f"Saved processed dataset to: {OUTPUT_PATH}")
