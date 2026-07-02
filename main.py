"""
Please modify the following paths according to your actual environment before running:
    - R_HOME               : R installation path
    - R_LIB_PATH           : directory where the mclust package is located
    - DATA_DIR             : data folder
    - OUTPUT_FILE          : output file save path
"""

import os
import torch
import pandas as pd
import scanpy as sc
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Custom modules
from graph_GCN import graph_GCN
from preprocess import pca, construct_neighbor_graph, get_mvdataSet, clr_normalize_each_cell, fix_seed
from train import SMGC
from utils import clustering

# ---------- Environment configuration ----------
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# R configuration (for mclust clustering)
os.environ['R_HOME'] = '/root/.conda/envs/envsg/lib/R'      
import rpy2.robjects as robjects
target_lib = "/root/.conda/envs/envsg/lib/R/library"          
robjects.r(f'.libPaths("{target_lib}")')
robjects.r.library("mclust")
print("mclust loaded successfully!")

# Fix random seed
random_seed = 2022
fix_seed(random_seed)

# ---------- Path settings ----------
DATA_DIR = '/root/shared-nvme/dzxdata/Human_Lymph_Nodes/D1/'  
OUTPUT_FILE = '/root/new/clustering-visual/HLN-D1_output_results.h5ad'  # output file

# ---------- 1. Load data ----------
adata_omics1 = sc.read_h5ad(DATA_DIR + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(DATA_DIR + 'adata_ADT.h5ad')

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

data_type = '10x'  

# ---------- 2. Preprocessing ----------
sc.pp.filter_genes(adata_omics1, min_cells=10)
sc.pp.highly_variable_genes(
    adata_omics1,
    flavor="seurat_v3",
    n_top_genes=3000
)
sc.pp.normalize_total(adata_omics1, target_sum=1e4)
sc.pp.log1p(adata_omics1)
sc.pp.scale(adata_omics1)

adata_omics1_high = adata_omics1[:, adata_omics1.var['highly_variable']]
adata_omics1.obsm['feat'] = pca(
    adata_omics1_high,
    n_comps=adata_omics2.n_vars - 1
)


adata_omics2 = clr_normalize_each_cell(adata_omics2)
sc.pp.scale(adata_omics2)
adata_omics2.obsm['feat'] = pca(
    adata_omics2,
    n_comps=adata_omics2.n_vars - 1
)

# ---------- 3. Construct neighbor graph ----------
data = construct_neighbor_graph(
    adata_omics1,
    adata_omics2,
    datatype=data_type
)

# ---------- 4. Initialize graph_GCN and generate representations ----------
model = graph_GCN(
    data=data,
    datatype=data_type,
    device=device,
    random_seed=random_seed,
    dim_input=3000,      # number of highly variable genes
    dim_output=128       # output representation dimension
)


output = model.generate_representations()

# Extract four types of representations
emb_spatial_omics1 = output['emb_latent_spatial_omics1']
emb_spatial_omics2 = output['emb_latent_spatial_omics2']
emb_feature_omics1 = output['emb_latent_feature_omics1']
emb_feature_omics2 = output['emb_latent_feature_omics2']

print(f"Modality 1 spatial representation: {emb_spatial_omics1.shape}")
print(f"Modality 2 spatial representation: {emb_spatial_omics2.shape}")
print(f"Modality 1 feature representation: {emb_feature_omics1.shape}")
print(f"Modality 2 feature representation: {emb_feature_omics2.shape}")

# Store representations into AnnData object
adata = adata_omics1.copy()
adata.obsm['emb_spatial_omics1'] = emb_spatial_omics1
adata.obsm['emb_spatial_omics2'] = emb_spatial_omics2
adata.obsm['emb_feature_omics1'] = emb_feature_omics1
adata.obsm['emb_feature_omics2'] = emb_feature_omics2


adata.obs['ground_truth'] = adata.obs['Spatial_Label']

# ---------- 5. Build multi-view dataset ----------
mv_dataset = get_mvdataSet(adata, device=device, normalize=True)

# ---------- 6. Train SMGC model ----------
lr = 1e-4
epochs = 100
latent_dim = 32
p = 5
batch_size = 64
use_linear_projection = False

trainer = SMGC(
    mv_dataset,
    device,
    lr,
    epochs,
    latent_dim,
    p,
    batch_size,
    use_linear_projection
)

final_features = trainer.train()
print(f"Training completed! Final feature shape: {final_features.shape}")

# Store fused features
adata.obsm['SMGC_emb'] = final_features

# ---------- 7. Clustering ----------
tool = 'mclust'
adata = clustering(
    adata,
    key='SMGC_emb',
    add_key='SMGC_cluster',
    n_clusters=10,
    method=tool,
    use_pca=True
)

print("Clustering result statistics:")
print(adata.obs['SMGC_cluster'].value_counts())

# ---------- 8. Evaluation ----------
true_labels = adata.obs['ground_truth']
pred_labels = adata.obs['SMGC_cluster']

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='max')
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")

# ---------- 9. Save results ----------
adata.write(OUTPUT_FILE)
print("Saved successfully!")
print(adata)

if __name__ == '__main__':

    pass  # all operations have been executed at the top level
