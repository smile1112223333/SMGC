"""
运行前请根据实际环境修改以下路径：
    - R_HOME               : R安装路径
    - R_LIB_PATH           : mclust包所在目录
    - DATA_DIR             : 数据文件夹
    - OUTPUT_FILE          : 输出文件保存路径
"""

import os
import torch
import pandas as pd
import scanpy as sc
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# 自定义模块
from graph_GCN import graph_GCN
from preprocess import pca, construct_neighbor_graph, get_mvdataSet, clr_normalize_each_cell, fix_seed
from train import SMGC
from utils import clustering

# ---------- 环境配置 ----------
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# R 配置（用于mclust聚类）
os.environ['R_HOME'] = '/root/.conda/envs/envsg/lib/R'      
import rpy2.robjects as robjects
target_lib = "/root/.conda/envs/envsg/lib/R/library"          
robjects.r(f'.libPaths("{target_lib}")')
robjects.r.library("mclust")
print("mclust 加载成功！")

# 固定随机种子
random_seed = 2022
fix_seed(random_seed)

# ---------- 路径设置 ----------
DATA_DIR = '/root/shared-nvme/dzxdata/Human_Lymph_Nodes/A1/'  
OUTPUT_FILE = '/root/new下游分析/clustering-visual/HLN-A1_output_results.h5ad'  # 输出文件

# ---------- 1. 读取数据 ----------
adata_omics1 = sc.read_h5ad(DATA_DIR + 'adata_RNA.h5ad')
adata_omics2 = sc.read_h5ad(DATA_DIR + 'adata_ADT.h5ad')

adata_omics1.var_names_make_unique()
adata_omics2.var_names_make_unique()

data_type = '10x'  

# ---------- 2. 预处理 ----------
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

# ---------- 3. 构建邻居图 ----------
data = construct_neighbor_graph(
    adata_omics1,
    adata_omics2,
    datatype=data_type
)

# ---------- 4. 初始化 graph_GCN 并生成表示 ----------
model = graph_GCN(
    data=data,
    datatype=data_type,
    device=device,
    random_seed=random_seed,
    dim_input=3000,      # 高变基因数
    dim_output=128       # 输出表示维度
)


output = model.generate_representations()

# 提取四种表示
emb_spatial_omics1 = output['emb_latent_spatial_omics1']
emb_spatial_omics2 = output['emb_latent_spatial_omics2']
emb_feature_omics1 = output['emb_latent_feature_omics1']
emb_feature_omics2 = output['emb_latent_feature_omics2']

print(f"模态1 空间表示: {emb_spatial_omics1.shape}")
print(f"模态2 空间表示: {emb_spatial_omics2.shape}")
print(f"模态1 特征表示: {emb_feature_omics1.shape}")
print(f"模态2 特征表示: {emb_feature_omics2.shape}")

# 将表示存入 AnnData 对象
adata = adata_omics1.copy()
adata.obsm['emb_spatial_omics1'] = emb_spatial_omics1
adata.obsm['emb_spatial_omics2'] = emb_spatial_omics2
adata.obsm['emb_feature_omics1'] = emb_feature_omics1
adata.obsm['emb_feature_omics2'] = emb_feature_omics2


adata.obs['ground_truth'] = adata.obs['Spatial_Label']

# ---------- 5. 构建多视图数据集 ----------
mv_dataset = get_mvdataSet(adata, device=device, normalize=True)

# ---------- 6. 训练 SMGC 模型 ----------
lr = 1e-4
epochs = 20
latent_dim = 64
p = 10
batch_size = 32
use_linear_projection = True

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
print(f"训练完成！最终特征形状: {final_features.shape}")

# 存储融合后的特征
adata.obsm['SMGC_emb'] = final_features

# ---------- 7. 聚类 ----------
tool = 'mclust'
adata = clustering(
    adata,
    key='SMGC_emb',
    add_key='SMGC_cluster',
    n_clusters=10,
    method=tool,
    use_pca=True
)

print("聚类结果统计：")
print(adata.obs['SMGC_cluster'].value_counts())

# ---------- 8. 评估 ----------
true_labels = adata.obs['ground_truth']
pred_labels = adata.obs['SMGC_cluster']

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method='max')
print(f"ARI: {ari:.4f}")
print(f"NMI: {nmi:.4f}")

# ---------- 9. 保存结果 ----------
adata.write(OUTPUT_FILE)
print("保存成功！")
print(adata)

if __name__ == '__main__':
    pass  # 所有操作已在顶层执行