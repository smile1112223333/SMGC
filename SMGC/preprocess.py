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
    # 
    labels = adata.obs['ground_truth']
    
    if labels.dtype == 'object' or isinstance(labels.iloc[0], str):
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)
        print(f"labels: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    else:
        labels = labels.values
    
    # 
    view_list = []
    
    # view1 Spatial Graph Representation of Modality 1
    view_1 = adata.obsm['emb_spatial_omics1']
    view_1 = torch.tensor(view_1, dtype=torch.float32) 
    view_list.append(view_1)
    
    # view2 Spatial Graph Representation of Modality 2
    view_2 = adata.obsm['emb_spatial_omics2']
    view_2 = torch.tensor(view_2, dtype=torch.float32)
    view_list.append(view_2)
    
    # view3 Feature Graph Representation of Modality 1
    view_3 = adata.obsm['emb_feature_omics1']
    view_3 = torch.tensor(view_3, dtype=torch.float32)
    view_list.append(view_3)
    
    # 视图4：Feature Graph Representation of Modality 2
    view_4 = adata.obsm['emb_feature_omics2']
    view_4 = torch.tensor(view_4, dtype=torch.float32)
    view_list.append(view_4)
    
    
    min_len = min([v.shape[0] for v in view_list])
    min_len_idx = torch.arange(min_len)
    view_list = [v[min_len_idx] for v in view_list]
    labels = labels[min_len_idx.cpu().numpy()]
    
    
    view_dims = []
    data = view_list  
    for i in range(len(data)): 
        if normalize:
            max_value, _ = torch.max(data[i], dim=0, keepdim=True)
            min_value, _ = torch.min(data[i], dim=0, keepdim=True)
           
            data[i] = (data[i] - min_value) / (max_value - min_value + 1e-12)
        
        view_dims.append(data[i].shape[1])  
    

    class MVDataset:
        def __init__(self, view_list, view_dims, labels):
            self.view_list = view_list 
            self.num_view = len(view_list) 
            self.view_dims = view_dims 
            self.labels = torch.tensor(labels, dtype=torch.int64).view(-1)  # CPU tensor
            self.num_class = len(torch.unique(self.labels))
            self.length = len(view_list[0]) if view_list else 0
        
        def __len__(self):
            return self.length
        
        def __getitem__(self, idx):
            return [view[idx] for view in self.view_list], self.labels[idx]
    
    mv_dataset = MVDataset(view_list, view_dims, labels)
    
    # 打印信息
    print(f"\n Dataset information：")
    print(f"- Number of samples: {len(mv_dataset)}")
    print(f"- Number of views: {mv_dataset.num_view}")
    print(f"- Dimension of each view: {mv_dataset.view_dims}")
    print(f"- Number of classes: {mv_dataset.num_class}")
    # print(f"- Label: {torch.bincount(mv_dataset.labels).cpu().numpy()}")
    
    return mv_dataset


def load_and_process_labels(adata, meta_path, sep='\t'):

    df = pd.read_csv(meta_path, sep=sep)

    if 'Joint_clusters' not in df.columns:
        raise ValueError(f"Column 'Joint_clusters' not found in the metadata file. Available columns: {df.columns.tolist()}")

    joint_clusters = df['Joint_clusters'].copy()
    
    if joint_clusters.isna().any():
        print(f"\nFound {joint_clusters.isna().sum()} missing values, marking them as 'Unknown'")
        joint_clusters = joint_clusters.fillna('Unknown')

    joint_clusters = joint_clusters.astype(str)
    

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(joint_clusters)
    

    for original, encoded in zip(label_encoder.classes_, range(len(label_encoder.classes_))):
        count = (joint_clusters == original).sum()
        print(f"  {original} -> {encoded} (number: {count})")
    

    if len(encoded_labels) != adata.n_obs:
        print(f"\nWarning: Number of labels ({len(encoded_labels)}) does not match the number of observations in adata ({adata.n_obs})!")
        # If the counts do not match, further processing may be required
        if len(encoded_labels) > adata.n_obs:
            print("Truncating labels to match the number of observations in adata")
            encoded_labels = encoded_labels[:adata.n_obs]
        else:
            print("Using the first N labels; the extra observations in adata will have no labels")
    else:
        print(f"\nLabels match the number of observations in adata: {len(encoded_labels)}")
    

    print(f"\n=== Processing completed ===")
    print(f"Final label distribution:\n{adata.obs['ground_truth_encoded'].value_counts().sort_index()}")
    print(f"Total number of classes: {len(np.unique(encoded_labels))}")
    return encoded_labels[:adata.n_obs], label_encoder
