#!/usr/bin/env python
"""
scRNA-seq pipeline (Scanpy): QC -> Scrublet -> normalization -> HVG -> scale/PCA ->
Harmony -> neighbors/UMAP -> Leiden -> marker-based annotation -> figures -> save.

Notes
- No patient/platform columns exist in this dataset's obs (only 'barcode'), so the script
  detects candidates defensively. If none are found, Scrublet runs on all cells together,
  HVGs are chosen without batch_key, and Harmony is skipped (fallback to uncorrected PCA).
- adata.raw holds ALL genes, log-normalized and unscaled. It is set after log1p and before
  HVG subsetting, so the HVG-subset object is the one that gets scaled and passed to PCA.
"""

import os
import gc
import re
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import scanpy as sc

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths and settings
# ----------------------------------------------------------------------------
INPUT_PATH = "/scratch/home/lfung/CS223_class/CS223_project/data/broad1.h5ad"
OUTPUT_DIR = "/scratch/home/lfung/CS223_class/CS223_project/output"
FIG_DIR = "/scratch/home/lfung/CS223_class/CS223_project/output/figures"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "broad1_annotated_v2.h5ad")
MARKERS_CSV = os.path.join(FIG_DIR, "cluster_markers.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

assert os.path.abspath(OUTPUT_PATH) != os.path.abspath(INPUT_PATH), "Refusing to overwrite input"

sc.settings.figdir = FIG_DIR
sc.settings.n_jobs = 4
sc.settings.verbosity = 2
sc.set_figure_params(dpi=100, dpi_save=150, frameon=False, figsize=(5, 5))


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Step 1: Load data
# ----------------------------------------------------------------------------
log("[1] Reading input data...")
adata = sc.read_h5ad(INPUT_PATH)
adata.var_names_make_unique()
adata.obs_names_make_unique()

# Keep sparse CSR float32 (memory efficient)
if not sp.issparse(adata.X):
    adata.X = sp.csr_matrix(adata.X)
adata.X = adata.X.tocsr().astype(np.float32)
log(f"    Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

# ----------------------------------------------------------------------------
# Step 2: Detect patient / platform columns (defensive)
# ----------------------------------------------------------------------------
def find_col(obs, patterns, max_levels=500):
    """Find a categorical-like obs column whose name matches one of the regex patterns."""
    for col in obs.columns:
        if any(re.search(p, col, flags=re.IGNORECASE) for p in patterns):
            n = obs[col].nunique()
            if 1 < n <= max_levels:
                return col
    return None


patient_col = find_col(adata.obs, [r"patient", r"donor", r"subject", r"sample", r"individual"])
platform_col = find_col(adata.obs, [r"platform", r"technology", r"chemistry", r"protocol", r"assay"])
log(f"    Patient column: {patient_col} | Platform column: {platform_col}")

# Batch key used for Scrublet / HVG (patient preferred)
batch_key = patient_col
for c in (patient_col, platform_col):
    if c is not None:
        adata.obs[c] = adata.obs[c].astype(str).astype("category")

# ----------------------------------------------------------------------------
# Step 3: QC metrics (mito, ribo, hemoglobin) with defensive prefix checks
# ----------------------------------------------------------------------------
log("[2] Calculating QC metrics...")
upper_names = adata.var_names.str.upper()
adata.var["mt"] = upper_names.str.startswith("MT-")  # covers 'MT-' and 'mt-'
adata.var["ribo"] = upper_names.str.startswith(("RPS", "RPL"))
adata.var["hb"] = upper_names.str.match(r"^HB(?!P)[A-Z0-9]*$") & ~upper_names.str.startswith("HBEGF")
log(f"    MT genes: {int(adata.var['mt'].sum())}, ribo genes: {int(adata.var['ribo'].sum())}, "
    f"Hb genes: {int(adata.var['hb'].sum())}")

sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], percent_top=None,
                           log1p=False, inplace=True)

# QC violin BEFORE filtering
try:
    sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                 jitter=False, multi_panel=True, show=False, save="_qc_before_filtering.png")
    plt.close("all")
except Exception as e:
    log(f"    WARNING: QC violin failed: {e}")

# ----------------------------------------------------------------------------
# Step 4: Filtering
# ----------------------------------------------------------------------------
log("[3] Filtering cells and genes...")
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
keep = (adata.obs["pct_counts_mt"] < 10).values
adata = adata[keep].copy()
gc.collect()
log(f"    After filtering: {adata.n_obs} cells x {adata.n_vars} genes")

# ----------------------------------------------------------------------------
# Step 5: Doublet detection (Scrublet) on RAW counts, per sample if available
# ----------------------------------------------------------------------------
log("[4] Running Scrublet on raw counts...")
try:
    if batch_key is not None:
        sc.pp.scrublet(adata, batch_key=batch_key)
    else:
        log("    No patient column; running Scrublet on all cells together.")
        sc.pp.scrublet(adata)

    # Doublet score histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(adata.obs["doublet_score"].values, bins=100, color="steelblue")
    ax.set_xlabel("Scrublet doublet score")
    ax.set_ylabel("Number of cells")
    ax.set_yscale("log")
    ax.set_title(f"Doublet scores (predicted doublets: {int(adata.obs['predicted_doublet'].sum())})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scrublet_doublet_score_hist.png"), dpi=150)
    plt.close(fig)

    n_dbl = int(adata.obs["predicted_doublet"].sum())
    adata = adata[~adata.obs["predicted_doublet"].astype(bool).values].copy()
    gc.collect()
    log(f"    Removed {n_dbl} predicted doublets; {adata.n_obs} cells remain")
except Exception as e:
    log(f"    WARNING: Scrublet failed ({e}); continuing without doublet removal.")
    if "doublet_score" not in adata.obs:
        adata.obs["doublet_score"] = np.float32(0.0)
    if "predicted_doublet" not in adata.obs:
        adata.obs["predicted_doublet"] = False
    plt.close("all")

# ----------------------------------------------------------------------------
# Step 6: Normalization + log1p
# ----------------------------------------------------------------------------
log("[5] Normalizing (10,000) and log1p...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.X = adata.X.astype(np.float32)

# ----------------------------------------------------------------------------
# Step 7: HVG selection (seurat flavor on log data)
# ----------------------------------------------------------------------------
log("[6] Selecting 2000 HVGs...")
try:
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat", batch_key=batch_key)
except Exception as e:
    log(f"    WARNING: HVG with batch_key failed ({e}); retrying without batch_key.")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")

try:
    sc.pl.highly_variable_genes(adata, show=False, save="_hvg.png")
    plt.close("all")
except Exception as e:
    log(f"    WARNING: HVG plot failed: {e}")

# ----------------------------------------------------------------------------
# Step 8: Attach all-gene log-normalized (unscaled) data as raw, then subset to HVGs
# ----------------------------------------------------------------------------
log("[7] Setting adata.raw (all genes, log-normalized) and subsetting to HVGs...")
adata.raw = adata  # all genes, log-normalized, unscaled
hvg_mask = adata.var["highly_variable"].values
adata = adata[:, hvg_mask].copy()  # old full object released after reassignment
gc.collect()
log(f"    HVG object: {adata.n_obs} cells x {adata.n_vars} genes")

# ----------------------------------------------------------------------------
# Step 9: Scale (on HVG subset) and PCA
# ----------------------------------------------------------------------------
log("[8] Scaling (clip 10) and PCA (50 comps)...")
sc.pp.scale(adata, max_value=10)
adata.X = adata.X.astype(np.float32)
sc.tl.pca(adata, n_comps=50, svd_solver="arpack")
try:
    sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, show=False, save="_pca_variance_ratio.png")
    plt.close("all")
except Exception as e:
    log(f"    WARNING: PCA variance plot failed: {e}")
gc.collect()

# ----------------------------------------------------------------------------
# Step 10: Harmony integration (fallback to uncorrected PCA)
# ----------------------------------------------------------------------------
log("[9] Batch correction with Harmony...")
use_rep = "X_pca"
harmony_key = None
if patient_col is not None and platform_col is not None:
    adata.obs["harmony_batch"] = (adata.obs[patient_col].astype(str) + "_" +
                                  adata.obs[platform_col].astype(str)).astype("category")
    harmony_key = "harmony_batch"
elif patient_col is not None:
    harmony_key = patient_col
elif platform_col is not None:
    harmony_key = platform_col

if harmony_key is None:
    log("    WARNING: no patient/platform column available; skipping Harmony, using uncorrected PCA.")
else:
    try:
        import scanpy.external as sce
        sce.pp.harmony_integrate(adata, key=harmony_key, basis="X_pca",
                                 adjusted_basis="X_pca_harmony")
        use_rep = "X_pca_harmony"
        log("    Harmony complete -> obsm['X_pca_harmony']")
    except Exception as e:
        log(f"    WARNING: Harmony failed or harmonypy missing ({e}); falling back to uncorrected PCA.")
        use_rep = "X_pca"

# ----------------------------------------------------------------------------
# Step 11: Neighbors, UMAP, Leiden
# (a pre-Harmony UMAP is skipped: a second UMAP on this many cells is very costly)
# ----------------------------------------------------------------------------
log(f"[10] Neighbors (30 PCs, k=15) and UMAP from {use_rep}...")
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep=use_rep)
sc.tl.umap(adata)

log("[11] Leiden clustering (res 0.5)...")
sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2,
             directed=False, key_added="leiden")
log(f"    Found {adata.obs['leiden'].nunique()} clusters")
gc.collect()

# ----------------------------------------------------------------------------
# Step 12: Marker-based annotation
# ----------------------------------------------------------------------------
log("[12] Scoring marker sets...")
marker_sets = {
    "Malignant epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7"],
    "CAFs": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "PDGFRB"],
    "CD8+ T cells": ["CD3E", "CD3D", "CD8A", "CD8B", "GZMB", "NKG7"],
    "Tregs": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2"],
    "TAMs": ["CD68", "CD163", "CSF1R", "MRC1", "C1QA", "APOE"],
    "Dendritic cells": ["CD1C", "CLEC9A", "FCER1A", "LILRA4", "CLEC10A", "IRF8"],
}
raw_genes = set(adata.raw.var_names)
present_markers = {}
score_cols = {}
for ct, genes in marker_sets.items():
    found = [g for g in genes if g in raw_genes]
    missing = [g for g in genes if g not in raw_genes]
    if missing:
        log(f"    {ct}: missing markers {missing}")
    if found:
        present_markers[ct] = found
        col = "score_" + re.sub(r"[^A-Za-z0-9]+", "_", ct)
        sc.tl.score_genes(adata, found, score_name=col, use_raw=True)
        score_cols[ct] = col

# Individual Treg-defining scores (FOXP3, IL2RA) to compare against CD8+ T score
extra_cols = {}
for g in ["FOXP3", "IL2RA"]:
    if g in raw_genes:
        col = f"score_gene_{g}"
        sc.tl.score_genes(adata, [g], score_name=col, use_raw=True)
        extra_cols[g] = col

log("[13] Assigning cell types to clusters...")
MIN_SCORE = 0.05      # below this best score -> Unassigned
TREG_MARGIN = 0.10    # FOXP3/IL2RA score must exceed CD8 score by this margin

cluster_scores = adata.obs.groupby("leiden", observed=True)[list(score_cols.values())].mean()
cluster_scores.columns = list(score_cols.keys())
extra_scores = (adata.obs.groupby("leiden", observed=True)[list(extra_cols.values())].mean()
                if extra_cols else pd.DataFrame(index=cluster_scores.index))

cluster_labels = {}
for cl in cluster_scores.index:
    if cluster_scores.shape[1] == 0:
        cluster_labels[cl] = "Unassigned"
        continue
    scores = cluster_scores.loc[cl].copy()
    best = scores.idxmax()
    if scores[best] < MIN_SCORE:
        cluster_labels[cl] = "Unassigned"
        continue
    # Resolve Treg vs CD8+ T overlap
    if best in ("Tregs", "CD8+ T cells") and "Tregs" in scores and "CD8+ T cells" in scores:
        cd8 = scores["CD8+ T cells"]
        treg_gene_scores = [extra_scores.loc[cl, c] for c in extra_scores.columns] if extra_scores.shape[1] else []
        treg_clear = bool(treg_gene_scores) and (max(treg_gene_scores) > cd8 + TREG_MARGIN)
        if treg_clear and scores["Tregs"] >= MIN_SCORE:
            best = "Tregs"
        else:
            # Tregs not clearly supported: choose CD8 if it is valid, else next best non-Treg
            non_treg = scores.drop("Tregs")
            best = non_treg.idxmax()
            if non_treg[best] < MIN_SCORE:
                cluster_labels[cl] = "Unassigned"
                continue
    cluster_labels[cl] = best

adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_labels).astype(str).astype("category")
log("    Cluster -> cell type:")
for cl, ct in cluster_labels.items():
    log(f"      {cl}: {ct}")
log(f"    Cell type counts:\n{adata.obs['cell_type'].value_counts().to_string()}")

# ----------------------------------------------------------------------------
# Step 13: Differential expression per cluster (Wilcoxon on raw) -> top 10 CSV
# ----------------------------------------------------------------------------
log("[14] Ranking genes per Leiden cluster (Wilcoxon, raw)...")
try:
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon", use_raw=True,
                            key_added="rank_genes_groups")
    tables = []
    for g in adata.obs["leiden"].cat.categories:
        df = sc.get.rank_genes_groups_df(adata, group=g, key="rank_genes_groups").head(10)
        df.insert(0, "cluster", g)
        df.insert(1, "cell_type", cluster_labels.get(g, "Unassigned"))
        df.insert(2, "rank", np.arange(1, len(df) + 1))
        tables.append(df)
    pd.concat(tables, ignore_index=True).to_csv(MARKERS_CSV, index=False)
    log(f"    Saved top-10 markers to {MARKERS_CSV}")
except Exception as e:
    log(f"    WARNING: rank_genes_groups failed: {e}")

# ----------------------------------------------------------------------------
# Step 14: Figures
# ----------------------------------------------------------------------------
log("[15] Saving figures...")
umap_keys = ["leiden", "cell_type"]
for c in (patient_col, platform_col):
    if c is not None and c not in umap_keys:
        umap_keys.append(c)

for k in umap_keys:
    try:
        sc.pl.umap(adata, color=k, show=False, save=f"_{k}_after_integration.png",
                   legend_loc="on data" if k == "leiden" else "right margin")
        plt.close("all")
    except Exception as e:
        log(f"    WARNING: UMAP plot for {k} failed: {e}")

try:
    dot_markers = {ct: g for ct, g in present_markers.items()}
    sc.pl.dotplot(adata, var_names=dot_markers, groupby="cell_type", use_raw=True,
                  standard_scale="var", show=False, save="_markers_by_cell_type.png")
    plt.close("all")
except Exception as e:
    log(f"    WARNING: dotplot failed: {e}")

# ----------------------------------------------------------------------------
# Step 15: Save final AnnData
# ----------------------------------------------------------------------------
log("[16] Saving final AnnData...")
# Ensure obs columns are written safely
for c in adata.obs.columns:
    if adata.obs[c].dtype == object:
        adata.obs[c] = adata.obs[c].astype(str)
if "predicted_doublet" in adata.obs:
    adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype(bool)

adata.write_h5ad(OUTPUT_PATH)
log(f"    Saved to {OUTPUT_PATH}")

del adata
gc.collect()
log("Pipeline finished.")
