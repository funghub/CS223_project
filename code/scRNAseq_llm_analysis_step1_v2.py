"""Generation flow: one Claude call, then execute the generated script.

Pipeline requested from Claude: QC + mito filter, Scrublet doublet removal,
normalization, HVG, PCA, Harmony integration (patient/platform), neighbors,
UMAP, Leiden, marker-based cell type annotation, figures, save .h5ad.

Progress and errors are displayed directly in the terminal.
"""

import gc
import os
from pathlib import Path
import re
import subprocess
import sys

import anndata as ad
from anthropic import Anthropic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = "./data/broad1.h5ad"
output_file_path = "./output/"
OUTPUT_NAME = "broad1_annotated_v2.h5ad"  # differs from the input name
GENERATED_SCRIPT = PROJECT_ROOT / "generated" / "cs223_llm_step1_scanpy_analysis_v2.py"


def analyze_scrna_with_claude(file_path: str, output_file_path: str):
    file_path = (PROJECT_ROOT / file_path).resolve().as_posix()
    output_dir = (PROJECT_ROOT / output_file_path).resolve()
    output_file_path = output_dir.as_posix()
    print(f"Reading disk-backed dataset metadata: {file_path}", flush=True)
    adata = ad.read_h5ad(file_path, backed="r")
    try:
        num_cells = adata.n_obs
        num_genes = adata.n_vars
        obs_columns = list(adata.obs.columns)
        var_columns = list(adata.var.columns)
        clusters_found = [
            col for col in obs_columns
            if any(name in col.lower() for name in ("cluster", "leiden", "louvain"))
        ]
        if clusters_found:
            primary_cluster_col = clusters_found[0]
            cluster_counts = adata.obs[primary_cluster_col].value_counts().to_dict()
            cluster_context = f"Detected clusters ({primary_cluster_col}): {cluster_counts}"
        else:
            cluster_context = "No pre-computed structural clusters found in metadata."

        # Candidate batch columns (patient / platform) with number of levels,
        # so Claude does not have to guess column names.
        patient_keys = ("patient", "donor", "subject", "individual", "sample")
        platform_keys = ("platform", "tech", "chemistry", "assay", "method", "batch", "protocol")
        patient_cols = {c: int(adata.obs[c].nunique()) for c in obs_columns
                        if any(k in c.lower() for k in patient_keys)}
        platform_cols = {c: int(adata.obs[c].nunique()) for c in obs_columns
                         if any(k in c.lower() for k in platform_keys)}
        batch_context = (f"Candidate patient columns (name: n_levels): {patient_cols or 'none found'}. "
                         f"Candidate platform columns (name: n_levels): {platform_cols or 'none found'}.")
    finally:
        adata.file.close()
    del adata
    gc.collect()
    print(f"Dataset metadata: {num_cells} cells x {num_genes} genes", flush=True)
    print(f"Batch column candidates: {batch_context}", flush=True)

    prompt = f"""
    Act as an expert bioinformatician and Python developer specializing in single-cell RNA sequencing (scRNA-seq) analysis.

    Dataset information:
    - Cells: {num_cells}, Genes: {num_genes}
    - Cell metadata columns (obs): {obs_columns}
    - Gene metadata columns (var): {var_columns}
    - Context: {cluster_context}
    - Batch info: {batch_context}
    - Tissue context: tumor microenvironment (cancer), so expect malignant epithelial cells, CAFs and immune cells.

    Using Scanpy, write one script that performs QC, doublet removal, normalization, integration across
    patients and platforms, dimensionality reduction, clustering, and cell type annotation.

    Requirements:
    1. Input: Read the dataset from "{file_path}".
    2. Quality control and filtering:
       - Calculate QC metrics (mitochondrial, ribosomal, hemoglobin) with defensive checks for 'MT-' / 'mt-' prefixes.
       - Filter cells with min_genes=200, genes with min_cells=3, and cells with pct_counts_mt < 10.
    3. Doublet removal (Scrublet):
       - Run on RAW COUNTS before normalization, per sample/patient (use sc.pp.scrublet with batch_key,
         or scrublet directly per batch). Choose the sample/patient column from the candidates above.
       - Store doublet_score and predicted_doublet in obs, then remove predicted doublets.
       - Wrap in try/except: if Scrublet fails, log a warning and continue, never crash the pipeline.
    4. Normalization: normalize_total to 10,000, then log1p.
    5. Feature selection: HVGs (2000), using batch_key (patient) if available with flavor="seurat".
    6. Scale (clip max_value=10) on the HVG-subset object, then PCA (50 comps).
    7. Batch correction and integration:
       - Use Harmony via scanpy.external.pp.harmony_integrate on X_pca. If both patient and platform
         columns exist, build one combined categorical column (patient + "_" + platform) and use it as the key.
         Write the result to obsm["X_pca_harmony"].
       - Compute neighbors (30 PCs, n_neighbors=15) and UMAP from X_pca_harmony (use_rep="X_pca_harmony").
       - If harmonypy is not installed or integration fails, log a warning and fall back to uncorrected PCA.
    8. Clustering: Leiden, resolution 0.5, flavor="igraph", n_iterations=2, directed=False, key "leiden".
    9. Cell type annotation (marker-based, no external downloads):
       - Score each cell with sc.tl.score_genes (use adata.raw, keep only markers present in the data):
         Malignant epithelial: EPCAM, KRT8, KRT18, KRT19, KRT7
         CAFs: COL1A1, COL1A2, DCN, LUM, FAP, PDGFRB
         CD8+ T cells: CD3E, CD3D, CD8A, CD8B, GZMB, NKG7
         Tregs: FOXP3, IL2RA, CTLA4, TIGIT, IKZF2
         TAMs: CD68, CD163, CSF1R, MRC1, C1QA, APOE
         Dendritic cells: CD1C, CLEC9A, FCER1A, LILRA4, CLEC10A, IRF8
       - Assign each Leiden cluster the cell type with the highest mean score; write to obs["cell_type"]
         (label clusters with a low best score or no markers as "Unassigned").
       - Because Tregs and CD8+ T cells overlap, assign Tregs only if FOXP3 or IL2RA scores clearly
         exceed the CD8+ T score in that cluster.
       - Also run sc.tl.rank_genes_groups on leiden (wilcoxon, using raw) and save the top 10 genes per
         cluster to "{output_file_path}/figures/cluster_markers.csv".
    10. Figures: use matplotlib Agg backend, set sc.settings.figdir = "{output_file_path}/figures", and save:
       - QC violin (n_genes_by_counts, total_counts, pct_counts_mt) BEFORE filtering
       - Scrublet doublet score histogram
       - HVG plot and PCA variance ratio
       - UMAP colored by leiden, cell_type, the patient column, and the platform column
         (one before and one after Harmony if feasible, otherwise after only)
       - Dotplot of the marker genes grouped by cell_type
    11. Output: Save the final AnnData to "{OUTPUT_NAME}" in "{output_file_path}" (exact filename).
    Do not overwrite the input file. Write with compression="gzip" (or "lzf" for faster writes).
    Before saving, set adata.X to the sparse log-normalized matrix from adata.raw (subset to the HVGs) 
    (such as: adata.X = adata.raw[:, adata.var_names].X.tocsr().astype(np.float32))
    instead of the dense scaled matrix, and cast obsp["distances"] and obsp["connectivities"] to float32.

    Computer limit:
    32 GB RAM. Use in-memory Scanpy while minimizing peak memory including temporary copies.
    Keep the sparse format and use float32. Do not duplicate the full counts matrix in a layer.
    Subset to HVGs before attaching all-gene log-normalized expression as adata.raw (set raw before scaling),
    so downstream differential expression can use unscaled data. Free objects with del and gc.collect().

    Output rules:
    - Clear comments for each step and print progress messages (flush=True).
    - Output ONLY the complete, self-contained Python script wrapped in ```python ... ``` without omitting any steps.
    """

    max_tokens = 16000
    print(f"Sending dataset context to Claude: model=claude-sonnet-5-5, max_tokens={max_tokens}, effort=medium", flush=True)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=max_tokens,
        output_config={"effort": "medium"},
        cache_control={"type": "ephemeral"},
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((block for block in response.content if block.type == "text"), None)
    analysis_strategy = text_block.text if text_block else "No text response generated"

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "Claude reached max_tokens. Increase the response token limit before "
            "retrying; truncated code will not be executed."
        )

    code_block_match = re.search(r"```(?:python)?\s*(.*?)\s*```", analysis_strategy, re.DOTALL)
    generated_code = code_block_match.group(1).strip() if code_block_match else analysis_strategy.strip()

    # Syntax guard: do not execute code that does not compile.
    compile(generated_code, str(GENERATED_SCRIPT), "exec")
    GENERATED_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    GENERATED_SCRIPT.write_text(generated_code + "\n", encoding="utf-8")
    print(f"Generated script saved: {GENERATED_SCRIPT}", flush=True)
    print(analysis_strategy, flush=True)

    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["MPLBACKEND"] = "Agg"  # no display on the cluster
    result = subprocess.run(
        [sys.executable, "-u", str(GENERATED_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
    )
    expected_output = output_dir / OUTPUT_NAME
    if result.returncode == 0:
        print(f"Analysis process finished. Expected output: {expected_output}", flush=True)
    else:
        print(f"Execution failed with return code {result.returncode}", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(analyze_scrna_with_claude(file_path, output_file_path))
