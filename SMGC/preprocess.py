import os
import scipy
import anndata
import sklearn
import torch
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph 

from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics


def construct_neighbor_graph(adata_omics1, adata_omics2, datatype='10x', n_neighbors=3): 
    """
    Construct neighbor graphs, including feature graph and spatial graph. 
    Feature graph is based expression data while spatial graph is based on cell/spot spatial coordinates.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    data : dict
        AnnData objects with preprossed data for different omics.

    """

    # construct spatial neighbor graphs
    ################# spatial graph #################
    if datatype in ['Stereo-CITE-seq', 'Spatial-epigenome-transcriptome']:
       n_neighbors=6 
    # omics1
    cell_position_omics1 = adata_omics1.obsm['spatial']
    adj_omics1 = construct_graph_by_coordinate(cell_position_omics1, n_neighbors=n_neighbors)
    adata_omics1.uns['adj_spatial'] = adj_omics1
    
    # omics2
    cell_position_omics2 = adata_omics2.obsm['spatial']
    adj_omics2 = construct_graph_by_coordinate(cell_position_omics2, n_neighbors=n_neighbors)
    adata_omics2.uns['adj_spatial'] = adj_omics2
    
    ################# feature graph #################
    feature_graph_omics1, feature_graph_omics2 = construct_graph_by_feature(adata_omics1, adata_omics2)
    adata_omics1.obsm['adj_feature'], adata_omics2.obsm['adj_feature'] = feature_graph_omics1, feature_graph_omics2
    
    data = {'adata_omics1': adata_omics1, 'adata_omics2': adata_omics2}
    
    return data

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)
    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def clr_normalize_each_cell(adata, inplace=True):
    
    """Normalize count vector for each cell, i.e. for each row of .X"""

    import numpy as np
    import scipy

    def seurat_clr(x):
        # TODO: support sparseness
        s = np.sum(np.log1p(x[x > 0]))
        exp = np.exp(s / len(x))
        return np.log1p(x / exp)

    if not inplace:
        adata = adata.copy()
    
    # apply to dense or sparse matrix, along axis. returns dense matrix
    adata.X = np.apply_along_axis(
        seurat_clr, 1, (adata.X.A if scipy.sparse.issparse(adata.X) else np.array(adata.X))
    )
    return adata     

def construct_graph_by_feature(adata_omics1, adata_omics2, k=20, mode= "connectivity", metric="correlation", include_self=False):
    
    """Constructing feature neighbor graph according to expresss profiles"""
    
    feature_graph_omics1=kneighbors_graph(adata_omics1.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)
    feature_graph_omics2=kneighbors_graph(adata_omics2.obsm['feat'], k, mode=mode, metric=metric, include_self=include_self)

    return feature_graph_omics1, feature_graph_omics2

def construct_graph_by_coordinate(cell_position, n_neighbors=3):
    #print('n_neighbor:', n_neighbors)
    """Constructing spatial neighbor graph according to spatial coordinates."""
    
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(cell_position)  
    _ , indices = nbrs.kneighbors(cell_position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    adj = pd.DataFrame(columns=['x', 'y', 'value'])
    adj['x'] = x
    adj['y'] = y
    adj['value'] = np.ones(x.size)
    return adj

def transform_adjacent_matrix(adjacent):
    n_spot = adjacent['x'].max() + 1
    adj = coo_matrix((adjacent['value'], (adjacent['x'], adjacent['y'])), shape=(n_spot, n_spot))
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# ====== Graph preprocessing
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

def adjacent_matrix_preprocessing(adata_omics1, adata_omics2):
    """Converting dense adjacent matrix to sparse adjacent matrix"""
    
    ######################################## construct spatial graph ########################################
    adj_spatial_omics1 = adata_omics1.uns['adj_spatial']
    adj_spatial_omics1 = transform_adjacent_matrix(adj_spatial_omics1)
    adj_spatial_omics2 = adata_omics2.uns['adj_spatial']
    adj_spatial_omics2 = transform_adjacent_matrix(adj_spatial_omics2)
    
    adj_spatial_omics1 = adj_spatial_omics1.toarray()   # To ensure that adjacent matrix is symmetric
    adj_spatial_omics2 = adj_spatial_omics2.toarray()
    
    adj_spatial_omics1 = adj_spatial_omics1 + adj_spatial_omics1.T
    adj_spatial_omics1 = np.where(adj_spatial_omics1>1, 1, adj_spatial_omics1)
    adj_spatial_omics2 = adj_spatial_omics2 + adj_spatial_omics2.T
    adj_spatial_omics2 = np.where(adj_spatial_omics2>1, 1, adj_spatial_omics2)
    
    # convert dense matrix to sparse matrix
    adj_spatial_omics1 = preprocess_graph(adj_spatial_omics1) # sparse adjacent matrix corresponding to spatial graph
    adj_spatial_omics2 = preprocess_graph(adj_spatial_omics2)
    
    ######################################## construct feature graph ########################################
    adj_feature_omics1 = torch.FloatTensor(adata_omics1.obsm['adj_feature'].copy().toarray())
    adj_feature_omics2 = torch.FloatTensor(adata_omics2.obsm['adj_feature'].copy().toarray())
    
    adj_feature_omics1 = adj_feature_omics1 + adj_feature_omics1.T
    adj_feature_omics1 = np.where(adj_feature_omics1>1, 1, adj_feature_omics1)
    adj_feature_omics2 = adj_feature_omics2 + adj_feature_omics2.T
    adj_feature_omics2 = np.where(adj_feature_omics2>1, 1, adj_feature_omics2)
    
    # convert dense matrix to sparse matrix
    adj_feature_omics1 = preprocess_graph(adj_feature_omics1) # sparse adjacent matrix corresponding to feature graph
    adj_feature_omics2 = preprocess_graph(adj_feature_omics2)
    
    adj = {'adj_spatial_omics1': adj_spatial_omics1,
           'adj_spatial_omics2': adj_spatial_omics2,
           'adj_feature_omics1': adj_feature_omics1,
           'adj_feature_omics2': adj_feature_omics2,
           }
    
    return adj

def lsi(
        adata: anndata.AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
       ) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)
    """
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    #X = adata_use.X
    X_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(X)
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    #adata.obsm["X_lsi"] = X_lsi
    adata.obsm["X_lsi"] = X_lsi[:,1:]

def tfidf(X):
    r"""
    TF-IDF normalization (following the Seurat v3 approach)
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf   

def fix_seed(seed):
    #seed = 2023
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'    


def get_mvdataSet(adata, device, normalize=True):
    # -------------------------- 1. 处理标签 --------------------------
    labels = adata.obs['ground_truth']
    
    if labels.dtype == 'object' or isinstance(labels.iloc[0], str):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        print(f"标签编码映射: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        labels = labels.values
    
    # -------------------------- 2. 加载4个视图（tensor转换逻辑不变，正确可行） --------------------------
    view_list = []
    
    # 视图1：模态1 空间图表示
    view_1 = adata.obsm['emb_spatial_omics1']
    view_1 = torch.tensor(view_1, dtype=torch.float32) 
    view_list.append(view_1)
    
    # 视图2：模态2 空间图表示
    view_2 = adata.obsm['emb_spatial_omics2']
    view_2 = torch.tensor(view_2, dtype=torch.float32)
    view_list.append(view_2)
    
    # 视图3：模态1 特征图表示
    view_3 = adata.obsm['emb_feature_omics1']
    view_3 = torch.tensor(view_3, dtype=torch.float32)
    view_list.append(view_3)
    
    # 视图4：模态2 特征图表示
    view_4 = adata.obsm['emb_feature_omics2']
    view_4 = torch.tensor(view_4, dtype=torch.float32)
    view_list.append(view_4)
    
    # -------------------------- 3. 对齐样本长度 --------------------------
    min_len = min([v.shape[0] for v in view_list])
    min_len_idx = torch.arange(min_len)
    view_list = [v[min_len_idx] for v in view_list]
    labels = labels[min_len_idx.cpu().numpy()]
    
    # -------------------------- 4. 归一化（按你的要求修改格式） --------------------------
    view_dims = []
    data = view_list  # 定义data变量，和你给的代码格式一致
    for i in range(len(data)):  # 循环变量改为data[i]
        if normalize:
            max_value, _ = torch.max(data[i], dim=0, keepdim=True)
            min_value, _ = torch.min(data[i], dim=0, keepdim=True)
            # 完全按你指定的公式：(data[i] - min_value) / (max_value - min_value + 1e-12)
            data[i] = (data[i] - min_value) / (max_value - min_value + 1e-12)
        
        view_dims.append(data[i].shape[1])  # 记录视图维度
    
    # -------------------------- 5. 数据集类 --------------------------
    # 正确的Dataset __init__（仅存CPU tensor）
    # 修正后的 MVDataset（在 get_mvdataSet 函数内部）
    class MVDataset:
        def __init__(self, view_list, view_dims, labels):
            # 所有视图都存CPU tensor（不提前移GPU！）
            self.view_list = view_list  # view_list 中的每个元素都是 CPU tensor
            self.num_view = len(view_list)  # 固定为4（之前已强制检查）
            self.view_dims = view_dims  # 每个视图的维度（如[128,128,128,128]）
            self.labels = torch.tensor(labels, dtype=torch.int64).view(-1)  # CPU tensor
            self.num_class = len(torch.unique(self.labels))
            self.length = len(view_list[0]) if view_list else 0
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            # 返回：[4个单样本CPU tensor] + 单样本CPU标签
            return [view[idx] for view in self.view_list], self.labels[idx]
    
    # -------------------------- 6. 创建数据集实例 --------------------------
    mv_dataset = MVDataset(view_list, view_dims, labels)
    
    # 打印信息
    print(f"\n数据集信息：")
    print(f"- 样本数量: {len(mv_dataset)}")
    print(f"- 视图数量: {mv_dataset.num_view}")
    print(f"- 每个视图维度: {mv_dataset.view_dims}")
    print(f"- 类别数量: {mv_dataset.num_class}")
    # print(f"- 标签分布: {torch.bincount(mv_dataset.labels).cpu().numpy()}")
    
    return mv_dataset


def load_and_process_labels(adata, meta_path, sep='\t'):
    """
    加载并处理标签数据
    
    参数:
    - adata: AnnData对象
    - meta_path: 元数据文件路径
    - sep: 文件分隔符
    
    返回:
    - 处理后的标签数组和标签编码器
    """
    
    # 1. 读取元数据文件
    print(f"正在读取元数据文件: {meta_path}")
    df = pd.read_csv(meta_path, sep=sep)
    
    # 2. 检查必要列是否存在
    if 'Joint_clusters' not in df.columns:
        raise ValueError(f"元数据文件中未找到 'Joint_clusters' 列。可用列: {df.columns.tolist()}")
    
    # 3. 提取标签列
    joint_clusters = df['Joint_clusters'].copy()
    
    # # 4. 检查标签数据质量
    # print("\n=== 标签数据统计 ===")
    # print(f"总样本数: {len(joint_clusters)}")
    # print(f"标签分布:\n{joint_clusters.value_counts(dropna=False)}")
    # print(f"NaN值数量: {joint_clusters.isna().sum()}")
    # print(f"唯一标签数量: {joint_clusters.nunique()}")
    
    # 5. 处理缺失值
    if joint_clusters.isna().any():
        print(f"\n发现 {joint_clusters.isna().sum()} 个缺失值，将其标记为 'Unknown'")
        joint_clusters = joint_clusters.fillna('Unknown')
    
    # 6. 确保标签为字符串类型
    joint_clusters = joint_clusters.astype(str)
    
    # 7. 标签编码
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(joint_clusters)
    
    # 8. 打印编码映射
    print(f"\n=== 标签编码映射 ===")
    for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        count = (joint_clusters == original).sum()
        print(f"  {original} -> {encoded} (数量: {count})")
    
    # 9. 验证数据对齐
    if len(encoded_labels) != adata.n_obs:
        print(f"\n警告: 标签数量({len(encoded_labels)})与adata观测数({adata.n_obs})不匹配!")
        # 如果数量不匹配，可能需要进一步处理
        if len(encoded_labels) > adata.n_obs:
            print("将截断标签以匹配adata观测数")
            encoded_labels = encoded_labels[:adata.n_obs]
        else:
            print("将使用前N个标签，adata的多余观测将没有标签")
    else:
        print(f"\n标签与adata观测数匹配: {len(encoded_labels)}")
    
    # 10. 将处理后的标签添加到adata
    adata.obs['ground_truth'] = joint_clusters.values[:adata.n_obs]
    adata.obs['ground_truth_encoded'] = encoded_labels[:adata.n_obs]
    
    print(f"\n=== 处理完成 ===")
    print(f"最终标签分布:\n{adata.obs['ground_truth_encoded'].value_counts().sort_index()}")
    print(f"总类别数: {len(np.unique(encoded_labels))}")
    
    return encoded_labels[:adata.n_obs], label_encoder
