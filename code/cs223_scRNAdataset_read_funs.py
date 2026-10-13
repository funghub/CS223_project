#NAME: cs223_scRNAdataset_read_funs.py

# IMPORTS
import os
import pandas as pd
import anndata as ad
from scipy.io import mmread

"""
See AnnData documentation at  
  https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html#conversion-to-dataframes
and
  https://www.sc-best-practices.org/introduction/fundamental-data-structures-and-frameworks/
"""

###
# Reading BROAD Institute scRNA files named matrix.mtx, barcodes.tsv, and features.tsv
# Sometimes an additional file named metadata.tsv is available that contains descriptive
# information about the dataset.
###
def load_broad_dataset(data_dir):
    """
    Reads uncompressed Broad Institute single-cell data into an AnnData object.
    """
    matrix_path = os.path.join(data_dir, 'matrix.mtx')
    barcodes_path = os.path.join(data_dir, 'barcodes.tsv')
    genes_path = os.path.join(data_dir, 'genes.tsv')

    if not os.path.exists(genes_path):
        genes_path = os.path.join(data_dir, 'features.tsv')

    print("Step 1: Reading the sparse matrix...")
    # FIX: Pass the string path directly to mmread.
    # SciPy will open, read into memory, and close the file safely internally.
    X_sparse = mmread(matrix_path)
    X = X_sparse.tocsr().T

    print("Step 2: Loading cell and gene metadata...")
    barcodes = pd.read_csv(barcodes_path, header=None, sep='\t', names=['barcode'])
    genes = pd.read_csv(genes_path, header=None, sep='\t')

    if genes.shape[1] >= 2:
        genes.columns = ['gene_ids', 'gene_symbols']
    else:
        genes.columns = ['gene_ids']

    # Step 3: Align dimensions and set indices
    barcodes.index = barcodes['barcode'].astype(str)

    if 'gene_symbols' in genes.columns:
        genes.index = genes['gene_symbols'].astype(str)
        genes.index = ad.utils.make_index_unique(genes.index)
    else:
        genes.index = genes['gene_ids'].astype(str)

    print(f"Step 4: Assembling AnnData with dimensions: {X.shape}")
    adata = ad.AnnData(X=X, obs=barcodes, var=genes)

    return adata


###
# Reading .h5ad scRNA type files.
###
def load_h5ad_dataset(data_dir):
    # Read the file into an AnnData object
    adata = ad.read_h5ad(data_dir)
    return adata




### MAKE SURE MODULE IS IMPORTED
if __name__ == "__main__":
    print("Reading Broad Institute single-cell data...")
    adata = load_broad_dataset("./data/broad/")
    print("adata", adata)
    print("Writing adata object to ./data/broad/broad1.h5ad")
    adata.write_h5ad("./data/broad/broad1.h5ad")
    print("Conveting adata to DataFrame")
    adatadf = adata.to_df()
    print("Writing converted DataFrame to ./data/broad/broad1.csv,  ./data/broad/broad1.excel, and ./data/broad/broad1.json")
    print("This can take several minutes to write each file")
    print("Writing to .csv")
    #adatadf.to_csv('./data/broad/broad1.csv', index=False)
    print("Writing to .excel")
    #adatadf.to_excel('./data/broad/broad1.excel', index=False)
    print("Writing to .json")
    #adatadf.to_json('./data/broad/broad1.json', orient='index')
    print("Printing adatadf.")
    print(adatadf)
    print()
    print(">>>>>>>>>>>>>>><<<<<<<<<<<<<<<<")
    print()

    adata1 = load_h5ad_dataset("./data/pbmc_sample.h5ad")
    print(adata1)
    print("Comveting adata1 to DataFrame")
    adatadf1 = adata1.to_df()
    print(adatadf1)
    print()
    print(">>>>> DONE <<<<<")
    #print("Module cs223_scRNAdataset_read_funs.py is intended to be imported and not executed.")






