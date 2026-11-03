"""Pre-validation generation flow: one Claude call, then execute the script.

Restores the behavior before smoke tests, repair calls, the execution-contract
system prompt, timeouts, and the memory watchdog were added. Metadata is still
read in backed mode, and the earlier truncation/syntax guards are retained.

Output names are isolated so this comparison does not overwrite the successful
generated script or its processed dataset.

Each run writes generation and analysis logs under output/test/logs/<run-id>.
"""

import gc
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime
from uuid import uuid4

import anndata as ad
from anthropic import Anthropic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = "./data/broad1.h5ad"
output_file_path = "./output/"
GENERATED_SCRIPT = PROJECT_ROOT / "generated" / "cs223_llm_step1_scanpy_analysis.py"


def analyze_scrna_with_claude(file_path: str, output_file_path: str):
    output_dir = (PROJECT_ROOT / output_file_path).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    log_dir = output_dir / "logs" / run_id
    log_dir.mkdir(parents=True)
    logger = logging.getLogger(f"scrna_generation_test.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_dir / "generation.log", encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    try:
        logger.info("Run logs: %s", log_dir)
        return _generate_and_execute(file_path, output_file_path, log_dir, logger)
    except BaseException:
        logger.exception("Generation or execution failed. See this traceback and analysis.log if present.")
        raise
    finally:
        for handler in (file_handler, console_handler):
            logger.removeHandler(handler)
            handler.close()


def execute_with_log(script_path, log_path, logger):
    """Tee child stdout/stderr to disk and the terminal; no retries or watchdog."""
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(
            [sys.executable, "-u", str(script_path)], cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=environment,
        )
        logger.info("Analysis started, PID %s. Combined stdout/stderr: %s", child.pid, log_path)
        try:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = child.wait()
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait()
            raise
        finally:
            child.stdout.close()
    seconds = round(time.monotonic() - started, 2)
    logger.info("Analysis exited with return code %s after %.2f seconds", returncode, seconds)
    return returncode, seconds


def _generate_and_execute(file_path: str, output_file_path: str, log_dir, logger):
    file_path = (PROJECT_ROOT / file_path).resolve().as_posix()
    output_dir = (PROJECT_ROOT / output_file_path).resolve()
    output_file_path = output_dir.as_posix()
    logger.info("Reading disk-backed dataset metadata: %s", file_path)
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
    finally:
        adata.file.close()
    del adata
    gc.collect()
    logger.info("Dataset metadata: %s cells x %s genes", num_cells, num_genes)

    prompt = f"""
    Act as an expert bioinformatician and Python developer specializing in single-cell RNA sequencing (scRNA-seq) analysis. 

    Dataset information:
    - Cells: {num_cells}, Genes: {num_genes}
    - Cell metadata columns (obs): {obs_columns}
    - Gene metadata columns (var): {var_columns}
    - Context: {cluster_context}

    Using the Scanpy library to perform a standard quality control (QC), filtering, normalization, dimensionality reduction, and clustering workflow on a single-cell dataset.    

    Here are the requirements for the script:
    1. Input: Read the single-cell dataset directly from "{file_path}".
    2. Quality Control & Filtering:
       - Calculate QC metrics (mitochondrial, ribosomal, and hemoglobin genes if applicable).
       - Filter out low-quality cells based on standard thresholds (e.g., min_genes=200, min_cells=3).
       - Filter cells based on mitochondrial content (e.g., keeping cells with < 5% or 10% mitochondrial counts).
    3. Normalization & Log Transformation:
       - Normalize total counts per cell to 10,000.
       - Log-transform the data (log1p).
    4. Feature Selection & Scale:
       - Identify Highly Variable Genes (HVGs).
       - Scale the data to unit variance and zero mean, clipping max values to 10.
    5. Dimensionality Reduction:
       - Run Principal Component Analysis (PCA).
       - Compute the neighborhood graph (using an appropriate number of PCs, like 30).
       - Run UMAP embedding.
    6. Clustering:
       - Cluster the cells using the Leiden algorithm (set a default resolution of 0.5).
    7. Output: Save the final, fully processed AnnData object to a new file named "processed_dataset.h5ad" in "{output_file_path}".

    computer limit:
    The computer has 32 GB total RAM. Use standard in-memory Scanpy while minimizing
    peak memory, including temporary copies. Preserve the existing sparse format
    unless conversion is necessary. Do not duplicate the full counts matrix in a
    layer. Subset to HVGs before attaching all-gene log-normalized expression as
    adata.raw, so HVG selection does not copy the full raw matrix.
    
    Please include:
    - Clear comments explaining each step.
    - Output ONLY the complete, self-contained Python script wrapped in ```python ... ``` without omitting any steps.
    - Proper handling of the raw data layer (`adata.raw`) before scaling so downstream differential expression/marker gene analysis can use unscaled data.
    - Defensive checks (e.g., verifying if mitochondrial genes exist in the dataset using 'MT-' or 'mt-' prefixes).
    
    """

    logger.info("Sending dataset context to Claude: model=claude-sonnet-5-5, max_tokens=8000, effort=medium")
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-5-5",
        max_tokens=8000,
        output_config={"effort": "medium"},
        cache_control={"type": "ephemeral"},
        messages=[{"role": "user", "content": prompt}],
    )
    usage = response.usage.model_dump()
    (log_dir / "usage.json").write_text(json.dumps(usage, indent=2), encoding="utf-8")
    logger.info("Claude stop reason: %s; cache written: %s; cache reused: %s; output tokens: %s",
                response.stop_reason, response.usage.cache_creation_input_tokens,
                response.usage.cache_read_input_tokens, response.usage.output_tokens)

    text_block = next((block for block in response.content if block.type == "text"), None)
    analysis_strategy = text_block.text if text_block else "No text response generated"
    (log_dir / "response.txt").write_text(analysis_strategy, encoding="utf-8")

    if response.stop_reason == "max_tokens":
        raise RuntimeError(
            "Claude reached max_tokens. Increase the response token limit before "
            f"retrying. Partial response saved in {log_dir / 'response.txt'}; it will not be executed."
        )

    code_block_match = re.search(r"```(?:python)?\s*(.*?)\s*```", analysis_strategy, re.DOTALL)
    generated_code = code_block_match.group(1).strip() if code_block_match else analysis_strategy.strip()

    # This guard existed before the smoke-test/repair changes.
    (log_dir / "generated_analysis.py").write_text(generated_code + "\n", encoding="utf-8")
    compile(generated_code, str(log_dir / "generated_analysis.py"), "exec")
    GENERATED_SCRIPT.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    GENERATED_SCRIPT.write_text(generated_code + "\n", encoding="utf-8")
    logger.info("Generated script saved: %s; exact code snapshot: %s", GENERATED_SCRIPT,
                log_dir / "generated_analysis.py")
    print(analysis_strategy, flush=True)

    # Original single execution: no smoke test, repair loop, or memory watchdog.
    returncode, seconds = execute_with_log(GENERATED_SCRIPT, log_dir / "analysis.log", logger)
    (log_dir / "execution.json").write_text(json.dumps({
        "returncode": returncode, "seconds": seconds,
        "script": str(GENERATED_SCRIPT), "input": file_path,
        "expected_output": str(output_dir / "processed_dataset.h5ad"),
        "output_exists": (output_dir / "processed_dataset.h5ad").is_file(),
    }, indent=2), encoding="utf-8")
    if returncode == 0:
        logger.info("Analysis process finished. Expected output: %s", output_dir / "processed_dataset.h5ad")
    else:
        logger.error("Execution failed with return code %s. Read %s", returncode, log_dir / "analysis.log")
    return returncode


if __name__ == "__main__":
    raise SystemExit(analyze_scrna_with_claude(file_path, output_file_path))
