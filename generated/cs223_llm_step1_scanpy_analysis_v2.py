import os
import gc
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Paths and settings
# ----------------------------------------------------------------------------
INPUT_PATH = "C:/Users/jimmy/Documents/GitHub/CS223_project/data/broad1.h5ad"
OUTPUT_DIR = "C:/Users/jimmy/Documents/GitHub/CS223_project/output"
FIG_DIR = "C:/Users/jimmy/Documents/GitHub/CS223_project/output/figures"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "broad1_annotated_v2.h5ad")
MARKER_CSV = os.path.join(FIG_DIR, "cluster_markers.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

sc.settings.figdir = FIG_DIR
sc.settings.verbosity = 2
sc.set_figure_params(dpi=80, frameon=False)

PATIENT_COL = "sample_id"   # candidate patient column
PLATFORM_COL = None         # no platform column found


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------
log("[1] Reading input data...")
adata = sc.read_h5ad(INPUT_PATH)
log(f"    Loaded: {adata.n_obs} cells x {adata.n_vars} genes")

# Keep matrix as float32 CSR (raw counts)
if not sp.issparse(adata.X):
    adata.X = sp.csr_matrix(adata.X)
adata.X = adata.X.tocsr().astype(np.float32)
adata.var_names_make_unique()

if PATIENT_COL not in adata.obs.columns:
    log("    WARNING: patient column missing; batch-aware steps will be skipped.")
    PATIENT_COL = None
else:
    adata.obs[PATIENT_COL] = adata.obs[PATIENT_COL].astype(str).astype("category")

# ----------------------------------------------------------------------------
# 2. QC metrics
# ----------------------------------------------------------------------------
log("[2] Calculating QC metrics...")
# Preserve original metadata QC columns
orig_qc = {}
for c in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]:
    if c in adata.obs.columns:
        orig_qc[c] = adata.obs[c].copy()
        adata.obs[c + "_original"] = adata.obs[c].copy()

gnames = adata.var_names.str.upper()
adata.var["mt"] = gnames.str.startswith("MT-")
adata.var["ribo"] = gnames.str.startswith(("RPS", "RPL"))
adata.var["hb"] = gnames.str.match(r"^HB[ABDEGMQZ]\d*$") & ~gnames.str.startswith("HBEGF")

n_mt = int(adata.var["mt"].sum())
n_ribo = int(adata.var["ribo"].sum())
n_hb = int(adata.var["hb"].sum())
log(f"    MT genes: {n_mt}, ribosomal: {n_ribo}, hemoglobin: {n_hb}")

qc_vars = [v for v in ["mt", "ribo", "hb"] if adata.var[v].sum() > 0]
has_mt = False
if qc_vars:
    tmp_metrics = sc.pp.calculate_qc_metrics(
        adata, qc_vars=qc_vars, percent_top=None, log1p=False, inplace=False
    )
    # tmp_metrics is (obs_df, var_df)
    obs_df = tmp_metrics[0]
    for v in qc_vars:
        col = f"pct_counts_{v}"
        if v == "mt":
            continue  # handled below
        adata.obs[col] = obs_df[col].values
    if n_mt > 0:
        adata.obs["pct_counts_mt"] = obs_df["pct_counts_mt"].values
        has_mt = True
        log("    Using computed pct_counts_mt.")
    del tmp_metrics, obs_df

if not has_mt:
    if "pct_counts_mt" in adata.obs.columns and adata.obs["pct_counts_mt"].notna().any():
        has_mt = True
        log("    No MT genes detected; using existing metadata pct_counts_mt.")
    else:
        log("    Mitochondrial QC unavailable; skipping MT filter.")

for c in ["n_genes_by_counts", "total_counts"]:
    if c not in adata.obs.columns:
        adata.obs[c] = (
            np.asarray((adata.X > 0).sum(1)).ravel() if c == "n_genes_by_counts"
            else np.asarray(adata.X.sum(1)).ravel()
        )

# QC violin BEFORE filtering
log("    Saving pre-filter QC violin...")
vkeys = ["n_genes_by_counts", "total_counts"] + (["pct_counts_mt"] if has_mt else [])
sc.pl.violin(adata, vkeys, jitter=0.4, multi_panel=True, show=False,
             save="_qc_before_filtering.png")
plt.close("all")

# ----------------------------------------------------------------------------
# 3. Filtering
# ----------------------------------------------------------------------------
log("[3] Filtering cells and genes...")
n0 = adata.n_obs
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
if has_mt:
    keep = (adata.obs["pct_counts_mt"] < 10).values
    adata = adata[keep].copy()
log(f"    Cells: {n0} -> {adata.n_obs}; genes: {adata.n_vars}")
gc.collect()

# ----------------------------------------------------------------------------
# 4. Doublet removal (Scrublet on raw counts, per sample)
# ----------------------------------------------------------------------------
log("[4] Running Scrublet per sample...")
try:
    if PATIENT_COL is not None:
        sc.pp.scrublet(adata, batch_key=PATIENT_COL)
    else:
        sc.pp.scrublet(adata)
    log(f"    Predicted doublets: {int(adata.obs['predicted_doublet'].sum())}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(adata.obs["doublet_score"].values, bins=100)
    ax.set_xlabel("Scrublet doublet score")
    ax.set_ylabel("Cells")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "scrublet_doublet_score_hist.png"), dpi=100)
    plt.close(fig)

    adata = adata[~adata.obs["predicted_doublet"].astype(bool).values].copy()
    log(f"    Cells after doublet removal: {adata.n_obs}")
except Exception as e:
    log(f"    WARNING: Scrublet failed ({e}); continuing without doublet removal.")
gc.collect()

# ----------------------------------------------------------------------------
# 5. Normalization
# ----------------------------------------------------------------------------
log("[5] Normalizing (10,000) and log1p...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.X = adata.X.astype(np.float32)

# ----------------------------------------------------------------------------
# 6. HVG selection
# ----------------------------------------------------------------------------
log("[6] Selecting 2000 HVGs...")
try:
    if PATIENT_COL is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat",
                                    batch_key=PATIENT_COL)
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
except Exception as e:
    log(f"    WARNING: batch-aware HVG failed ({e}); retrying without batch_key.")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")

sc.pl.highly_variable_genes(adata, show=False, save="_hvg.png")
plt.close("all")

# Preserve all-gene log-normalized expression (exact order required)
all_genes = adata
adata = all_genes[:, all_genes.var["highly_variable"]].copy()
adata.raw = all_genes
del all_genes
gc.collect()
log(f"    HVG object: {adata.n_obs} x {adata.n_vars}")

# ----------------------------------------------------------------------------
# 7. Scale + PCA
# ----------------------------------------------------------------------------
log("[7] Scaling and PCA...")
sc.pp.scale(adata, max_value=10)
if sp.issparse(adata.X):
    adata.X = adata.X.toarray()
adata.X = np.asarray(adata.X, dtype=np.float32)  # cast back to float32
gc.collect()

sc.tl.pca(adata, n_comps=50, svd_solver="arpack" if False else "randomized")
adata.obsm["X_pca"] = adata.obsm["X_pca"].astype(np.float32)
sc.pl.pca_variance_ratio(adata, n_pcs=50, log=True, show=False, save="_pca_variance_ratio.png")
plt.close("all")

# ----------------------------------------------------------------------------
# 8. Harmony integration, neighbors, UMAP
# ----------------------------------------------------------------------------
log("[8] Batch correction and embedding...")
if PATIENT_COL is not None and PLATFORM_COL is not None:
    adata.obs["batch_key_combined"] = (
        adata.obs[PATIENT_COL].astype(str) + "_" + adata.obs[PLATFORM_COL].astype(str)
    ).astype("category")
    batch_key = "batch_key_combined"
elif PATIENT_COL is not None:
    batch_key = PATIENT_COL
else:
    batch_key = None

color_cols = [c for c in [PATIENT_COL, PLATFORM_COL] if c is not None]

# UMAP before Harmony (uncorrected PCA)
try:
    log("    Computing pre-Harmony UMAP...")
    sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep="X_pca", key_added="neighbors_uncorrected")
    sc.tl.umap(adata, neighbors_key="neighbors_uncorrected")
    if color_cols:
        sc.pl.umap(adata, color=color_cols, show=False, save="_before_harmony.png")
        plt.close("all")
    # Remove temporary uncorrected graph to save memory
    for k in ["neighbors_uncorrected_distances", "neighbors_uncorrected_connectivities"]:
        if k in adata.obsp:
            del adata.obsp[k]
    adata.uns.pop("neighbors_uncorrected", None)
    adata.obsm.pop("X_umap", None)
except Exception as e:
    log(f"    WARNING: pre-Harmony UMAP failed ({e}).")
gc.collect()

use_rep = "X_pca"
if batch_key is not None:
    try:
        import harmonypy  # noqa: F401
        import scanpy.external as sce
        sce.pp.harmony_integrate(adata, key=batch_key, basis="X_pca",
                                 adjusted_basis="X_pca_harmony")
        adata.obsm["X_pca_harmony"] = np.asarray(adata.obsm["X_pca_harmony"], dtype=np.float32)
        use_rep = "X_pca_harmony"
        log("    Harmony integration complete.")
    except Exception as e:
        log(f"    WARNING: Harmony unavailable/failed ({e}); using uncorrected PCA.")
else:
    log("    No batch key; skipping Harmony.")

sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30, use_rep=use_rep)
sc.tl.umap(adata)

# ----------------------------------------------------------------------------
# 9. Leiden clustering
# ----------------------------------------------------------------------------
log("[9] Leiden clustering...")
sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2,
             directed=False, key_added="leiden")
log(f"    Clusters: {adata.obs['leiden'].nunique()}")

# ----------------------------------------------------------------------------
# 10. Marker-based annotation
# ----------------------------------------------------------------------------
log("[10] Cell type annotation...")
marker_sets = {
    "Malignant epithelial": ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7"],
    "CAFs": ["COL1A1", "COL1A2", "DCN", "LUM", "FAP", "PDGFRB"],
    "CD8+ T cells": ["CD3E", "CD3D", "CD8A", "CD8B", "GZMB", "NKG7"],
    "Tregs": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2"],
    "TAMs": ["CD68", "CD163", "CSF1R", "MRC1", "C1QA", "APOE"],
    "Dendritic cells": ["CD1C", "CLEC9A", "FCER1A", "LILRA4", "CLEC10A", "IRF8"],
}
raw_names = set(adata.raw.var_names)
present = {ct: [g for g in gl if g in raw_names] for ct, gl in marker_sets.items()}

score_cols = {}
for ct, genes in present.items():
    log(f"    {ct}: {len(genes)}/{len(marker_sets[ct])} markers present")
    if genes:
        col = "score_" + ct.replace(" ", "_").replace("+", "pos")
        sc.tl.score_genes(adata, gene_list=genes, score_name=col, use_raw=True)
        score_cols[ct] = col

# Per-gene marker expression for Treg rule (FOXP3 / IL2RA)
def raw_gene_vec(g):
    if g not in raw_names:
        return None
    x = adata.raw[:, g].X
    x = x.toarray().ravel() if sp.issparse(x) else np.asarray(x).ravel()
    return x

foxp3 = raw_gene_vec("FOXP3")
il2ra = raw_gene_vec("IL2RA")

MIN_SCORE = 0.05       # below this best score -> Unassigned
TREG_MARGIN = 0.1      # Treg must clearly exceed CD8 score

cluster_to_type = {}
if score_cols:
    clusters = adata.obs["leiden"].astype(str).values
    mean_scores = pd.DataFrame(
        {ct: adata.obs[col].groupby(adata.obs["leiden"].values).mean()
         for ct, col in score_cols.items()}
    )
    log("    Mean scores per cluster:\n" + mean_scores.round(3).to_string())

    for cl in mean_scores.index:
        row = mean_scores.loc[cl]
        best = row.idxmax()
        if row[best] < MIN_SCORE:
            cluster_to_type[cl] = "Unassigned"
            continue
        if best == "Tregs" and "CD8+ T cells" in row.index:
            # Require FOXP3 or IL2RA to clearly exceed CD8+ T score
            mask = clusters == str(cl)
            ok = False
            cd8 = row["CD8+ T cells"]
            for vec in (foxp3, il2ra):
                if vec is not None and vec[mask].mean() > cd8 + TREG_MARGIN * 0 and \
                        (vec[mask].mean() - cd8) > TREG_MARGIN:
                    ok = True
            if not ok:
                rest = row.drop("Tregs")
                best = rest.idxmax()
                if rest[best] < MIN_SCORE:
                    best = "Unassigned"
        cluster_to_type[cl] = best
else:
    log("    No markers present; all cells Unassigned.")

adata.obs["cell_type"] = (
    adata.obs["leiden"].astype(str).map(cluster_to_type).fillna("Unassigned").astype("category")
)
log("    Cell type counts:\n" + adata.obs["cell_type"].value_counts().to_string())

# Cluster marker genes (Wilcoxon on raw)
log("    Ranking genes per cluster (Wilcoxon)...")
try:
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon",
                            use_raw=True, n_genes=50)
    res = adata.uns["rank_genes_groups"]
    rows = []
    for cl in res["names"].dtype.names:
        for rank in range(10):
            rows.append({
                "cluster": cl,
                "rank": rank + 1,
                "gene": res["names"][cl][rank],
                "score": res["scores"][cl][rank],
                "logfoldchange": res["logfoldchanges"][cl][rank],
                "pval_adj": res["pvals_adj"][cl][rank],
            })
    pd.DataFrame(rows).to_csv(MARKER_CSV, index=False)
    log(f"    Saved markers to {MARKER_CSV}")
except Exception as e:
    log(f"    WARNING: rank_genes_groups failed ({e}).")

# ----------------------------------------------------------------------------
# 11. Figures
# ----------------------------------------------------------------------------
log("[11] Saving figures...")
umap_colors = ["leiden", "cell_type"] + color_cols
sc.pl.umap(adata, color=umap_colors, show=False, save="_after_harmony.png")
plt.close("all")
for c in umap_colors:
    try:
        sc.pl.umap(adata, color=c, show=False, save=f"_{c}.png")
        plt.close("all")
    except Exception as e:
        log(f"    WARNING: UMAP for {c} failed ({e})")

all_markers = [g for gl in present.values() for g in gl]
all_markers = list(dict.fromkeys(all_markers))
if all_markers:
    try:
        sc.pl.dotplot(adata, var_names=all_markers, groupby="cell_type", use_raw=True,
                      show=False, save="_markers_by_cell_type.png")
        plt.close("all")
    except Exception as e:
        log(f"    WARNING: dotplot failed ({e})")

# ----------------------------------------------------------------------------
# 12. Save final object
# ----------------------------------------------------------------------------
log("[12] Preparing and saving final AnnData...")
# Replace dense scaled X with sparse log-normalized HVG matrix from raw
adata.X = adata.raw[:, adata.var_names].X.tocsr().astype(np.float32)

for k in ["distances", "connectivities"]:
    if k in adata.obsp:
        adata.obsp[k] = adata.obsp[k].astype(np.float32)

gc.collect()
adata.write_h5ad(OUTPUT_PATH, compression="gzip")
log(f"Done. Saved to {OUTPUT_PATH}")
