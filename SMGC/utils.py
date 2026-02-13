import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
from preprocess import pca
import matplotlib.pyplot as plt

import random
import torch
import json
import hdf5storage as hdf
import itertools


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='SMGC', random_seed=123):
    """
    Stable Mclust clustering via rpy2.
    This version ensures:
      - numpy → R matrix conversion
      - clears dimnames
      - assigns proper column names to avoid Mclust errors
    """
    import numpy as np
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr

    # activate numpy → R conversion
    rpy2.robjects.numpy2ri.activate()

    # load mclust
    mclust = importr('mclust')

    # set random seed in R
    ro.r['set.seed'](random_seed)

    # get data
    X = adata.obsm[used_obsm]
    r_mat = ro.numpy2ri.numpy2rpy(X)

    # clear R environment (avoid residual variables)
    ro.r('rm(list=ls())')

    # assign matrix to R, clear dimnames, enforce numeric matrix
    ro.r.assign('X', r_mat)
    ro.r('X <- as.matrix(X)')
    ro.r('storage.mode(X) <- "double"')
    ro.r('dimnames(X) <- NULL')

    # assign safe column names to avoid Mclust internal dimnames error
    ro.r(f'colnames(X) <- paste0("V", 1:{X.shape[1]})')
    ro.r('rownames(X) <- NULL')

    # optional: print check
    # print(ro.r('dim(X)'))
    # print(ro.r('colnames(X)'))

    # call Mclust
    try:
        res = ro.r(f'Mclust(X, G={num_cluster}, modelNames="{modelNames}")')
    except Exception as e:
        print("❌ Mclust报错:", e)
        raise

    # extract classification labels
    labels = np.array(res.rx2('classification'))
    adata.obs['mclust'] = labels.astype(int).astype(str)

    return adata




def clustering(adata, n_clusters=7, key='emb', add_key='SMGC', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    """\
    Spatial clustering based the latent representation.

    Parameters
    ----------
    adata : anndata
        AnnData object of scanpy package.
    n_clusters : int, optional
        The number of clusters. The default is 7.
    key : string, optional
        The key of the input representation in adata.obsm. The default is 'emb'.
    method : string, optional
        The tool for clustering. Supported tools include 'mclust', 'leiden', and 'louvain'. The default is 'mclust'. 
    start : float
        The start value for searching. The default is 0.1. Only works if the clustering method is 'leiden' or 'louvain'.
    end : float 
        The end value for searching. The default is 3.0. Only works if the clustering method is 'leiden' or 'louvain'.
    increment : float
        The step size to increase. The default is 0.01. Only works if the clustering method is 'leiden' or 'louvain'.  
    use_pca : bool, optional
        Whether use pca for dimension reduction. The default is false.

    Returns
    -------
    None.

    """
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']

    return adata

       
def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    '''\
    Searching corresponding resolution according to given cluster number
    
    Parameters
    ----------
    adata : anndata
        AnnData object of spatial data.
    n_clusters : int
        Targetting number of clusters.
    method : string
        Tool for clustering. Supported tools include 'leiden' and 'louvain'. The default is 'leiden'.    
    use_rep : string
        The indicated representation for clustering.
    start : float
        The start value for searching.
    end : float 
        The end value for searching.
    increment : float
        The step size to increase.
        
    Returns
    -------
    res : float
        Resolution.
        
    '''
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     

def plot_weight_value(alpha, label, modality1='mRNA', modality2='protein'):
  """\
  Plotting weight values
  
  """  
  import pandas as pd  
  
  df = pd.DataFrame(columns=[modality1, modality2, 'label'])  
  df[modality1], df[modality2] = alpha[:, 0], alpha[:, 1]
  df['label'] = label
  df = df.set_index('label').stack().reset_index()
  df.columns = ['label_SpatialGlue', 'Modality', 'Weight value']
  ax = sns.violinplot(data=df, x='label_SpatialGlue', y='Weight value', hue="Modality",
                split=True, inner="quart", linewidth=1, show=False)
  ax.set_title(modality1 + ' vs ' + modality2) 

  plt.tight_layout(w_pad=0.05)
  plt.show()     



def init_torch(seed):
    # 随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_json(path,  encoding='utf-8'):
    with open(path, "r", encoding=encoding) as fp:
        params = json.load(fp)
    return params


def load_mat(path, views=None, key_feature="data", key_label="labels"):
    data = hdf.loadmat(path)
    feature = []
    num_view = len(data[key_feature])
    label = data[key_label].reshape((-1,))
    num_smp = label.size
    for v in range(num_view):
        tmp = data[key_feature][v][0].squeeze()
        feature.append(tmp)
    # 打乱样本
    rand_permute = np.random.permutation(num_smp)
    for v in range(num_view):
        feature[v] = feature[v][rand_permute]
    label = label[rand_permute]
    if views is None or len(views) == 0:
        views = list(range(num_view))
    views_feature = [feature[v] for v in views]
    return views_feature, label


# 实现功能：返回参数组合的笛卡尔集
def get_all_parameters(*parameters_range):
    return itertools.product(*parameters_range)

