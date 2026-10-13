# NAME: cs223_build_scBubbleTree_example5.py

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# 1. Create a mock AnnData dataset representing standard single-cell metrics
np.random.seed(42)
n_cells = 800
n_genes = 100

# Generate a sparse expression matrix
counts = np.random.negative_binomial(n=10, p=0.8, size=(n_cells, n_genes))
adata = ad.AnnData(X=counts)
adata.obs_names = [f"Cell_{i}" for i in range(n_cells)]
adata.var_names = [f"Gene_{j}" for j in range(n_genes)]

# Pre-calculate typical quality control (QC) observations
adata.obs['total_counts'] = np.sum(adata.X, axis=1)                  # Cell total count
adata.obs['n_genes_by_counts'] = np.sum(adata.X > 0, axis=1)          # Gene count per cell

# 2. Group cells into biological clusters
n_clusters = 6
clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
adata.obs['leiden_clusters'] = clustering.fit_predict(adata.X).astype(str)

# 3. Calculate cluster centroids and aggregate attributes
cluster_stats = []
unique_clusters = sorted(adata.obs['leiden_clusters'].unique())

for idx, cluster in enumerate(unique_clusters):
    cluster_mask = adata.obs['leiden_clusters'] == cluster
    cluster_cells = adata.X[cluster_mask]

    # Quantitative bubble tracking
    cell_count = cluster_mask.sum()
    mean_cell_count = adata.obs.loc[cluster_mask, 'total_counts'].mean()
    mean_gene_count = adata.obs.loc[cluster_mask, 'n_genes_by_counts'].mean()
    centroid_profile = np.mean(cluster_cells, axis=0)

    cluster_stats.append({
        'cluster': cluster,
        'cell_count': cell_count,
        'mean_cell_count': mean_cell_count,
        'mean_gene_count': mean_gene_count,
        'profile': centroid_profile
    })

df_stats = pd.DataFrame(cluster_stats)
profiles_matrix = np.array(df_stats['profile'].tolist())

# 4. Generate the hierarchical linkage matrix among clusters
Z = linkage(profiles_matrix, method='ward')

# 5. Plot the scBubbletree
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the background dendrogram structure
dend = dendrogram(
    Z,
    labels=df_stats['cluster'].values,
    ax=ax,
    link_color_func=lambda k: '#4A5568',
    no_plot=False
)

# Extract coordinates where the tree branches terminate (leaves)
# Leaves are ordered according to the dendrogram's visual representation
leaf_order = dend['leaves']
icoord = np.array(dend['icoord'])
dcoord = np.array(dend['dcoord'])

# Extract coordinates for the leaf points at the bottom (dcoord == 0)
x_coords = []
for i in range(len(leaf_order)):
    # Find positions where the line meets the bottom axis
    matched_x = icoord[np.where(dcoord == 0)[0]]
    # Aggregate or isolate coordinates for clean visualization matching order
    # Default visual placement maps cleanly over equidistant increments of 10
    x_coords.append(10 * i + 5)

# Align stats to match the layout order of the leaves
ordered_stats = df_stats.iloc[leaf_order].reset_index(drop=True)

# Main Bubble Scaling Logic (bubble sizes represent total cells in cluster)
max_bubble_size = 3500
min_bubble_size = 400
cell_counts = ordered_stats['cell_count'].values
scaled_sizes = min_bubble_size + (cell_counts - cell_counts.min()) / (cell_counts.max() - cell_counts.min() or 1) * \
            (max_bubble_size - min_bubble_size)

# Color bubbles based on Mean Gene Count per Cell metric
scatter = ax.scatter(
    x_coords,
    np.zeros(len(x_coords)),
    s=scaled_sizes,
    c=ordered_stats['mean_gene_count'],
    cmap='viridis',
    edgecolors='black',
    linewidths=1.5,
    zorder=3
)

# Annotate each bubble leaf with text showing the Cluster ID and Cell Volume
for i, x in enumerate(x_coords):
    label_text = f"C{ordered_stats.loc[i, 'cluster']}\nn={ordered_stats.loc[i, 'cell_count']}"
    ax.text(x, -0.15, label_text, ha='center', va='top', fontsize=9, fontweight='bold')

# Layout presentation clean-up
cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.18, shrink=0.6)
cbar.set_label('Mean Distinct Genes Detected per Cell', fontsize=11)

ax.set_title('Hierarchical scBubbletree Implementation via AnnData', fontsize=14, pad=15)
ax.set_ylabel('Transcriptional Node Distance', fontsize=12)
ax.set_xticks([])  # Leaf clusters cleanly annotated by label loops
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.show()
