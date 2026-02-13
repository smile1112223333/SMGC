import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from tqdm import tqdm
from preprocess import adjacent_matrix_preprocessing

class Encoder_overall(Module):
    """\
    Overall encoder.

    Parameters
    ----------
    dim_in_feat_omics1 : int
        Dimension of input features for omics1.
    dim_in_feat_omics2 : int
        Dimension of input features for omics2. 
    dim_out_feat_omics1 : int
        Dimension of latent representation for omics1.
    dim_out_feat_omics2 : int
        Dimension of latent representation for omics2, which is the same as omics1.
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    results: a dictionary including four graph-specific representations.
    """
     
    def __init__(self, dim_in_feat_omics1, dim_out_feat_omics1, dim_in_feat_omics2, dim_out_feat_omics2, dropout=0.0, act=F.relu):
        super(Encoder_overall, self).__init__()
        self.dim_in_feat_omics1 = dim_in_feat_omics1
        self.dim_in_feat_omics2 = dim_in_feat_omics2
        self.dim_out_feat_omics1 = dim_out_feat_omics1
        self.dim_out_feat_omics2 = dim_out_feat_omics2
        self.dropout = dropout
        self.act = act
        
        self.encoder_omics1 = Encoder(self.dim_in_feat_omics1, self.dim_out_feat_omics1)
        self.encoder_omics2 = Encoder(self.dim_in_feat_omics2, self.dim_out_feat_omics2)
        
    def forward(self, features_omics1, features_omics2, adj_spatial_omics1, adj_feature_omics1, adj_spatial_omics2, adj_feature_omics2):
        # graph1 - 空间图编码
        emb_latent_spatial_omics1 = self.encoder_omics1(features_omics1, adj_spatial_omics1)  
        emb_latent_spatial_omics2 = self.encoder_omics2(features_omics2, adj_spatial_omics2)
        
        # graph2 - 特征图编码
        emb_latent_feature_omics1 = self.encoder_omics1(features_omics1, adj_feature_omics1)
        emb_latent_feature_omics2 = self.encoder_omics2(features_omics2, adj_feature_omics2)

        # 返回四个Graph-specific representation
        results = {
            'emb_latent_spatial_omics1': emb_latent_spatial_omics1,
            'emb_latent_spatial_omics2': emb_latent_spatial_omics2,
            'emb_latent_feature_omics1': emb_latent_feature_omics1,
            'emb_latent_feature_omics2': emb_latent_feature_omics2
        }
        
        return results

class Encoder(Module): 
    """\
    Modality-specific GNN encoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Latent representation.
    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Encoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act

        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x
    
class Decoder(Module):
    """\
    Modality-specific GNN decoder.

    Parameters
    ----------
    in_feat: int
        Dimension of input features.
    out_feat: int
        Dimension of output features. 
    dropout: int
        Dropout probability of latent representations.
    act: Activation function. By default, we use ReLU.    

    Returns
    -------
    Reconstructed representation.
    """
    
    def __init__(self, in_feat, out_feat, dropout=0.0, act=F.relu):
        super(Decoder, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dropout = dropout
        self.act = act
        
        self.weight = Parameter(torch.FloatTensor(self.in_feat, self.out_feat))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, feat, adj):
        x = torch.mm(feat, self.weight)
        x = torch.spmm(adj, x)
        return x

class graph_GCN:
    """\
    图GCN模型，只生成表示不进行优化
    """
    
    def __init__(self, 
        data,
        datatype='10x',
        device=torch.device('cpu'),
        random_seed=2025,
        dim_input=3000,
        dim_output=64
        ):
        '''\
        初始化图GCN模型

        Parameters
        ----------
        data : dict
            dict object of spatial multi-omics data.
        datatype : string, optional
            Data type of input.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 2022.    
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        '''
        self.data = data.copy()
        self.datatype = datatype
        self.device = device
        self.random_seed = random_seed
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        # 设置随机种子
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
        
        # 邻接矩阵
        self.adata_omics1 = self.data['adata_omics1']
        self.adata_omics2 = self.data['adata_omics2']
        self.adj = adjacent_matrix_preprocessing(self.adata_omics1, self.adata_omics2)
        self.adj_spatial_omics1 = self.adj['adj_spatial_omics1'].to(self.device)
        self.adj_spatial_omics2 = self.adj['adj_spatial_omics2'].to(self.device)
        self.adj_feature_omics1 = self.adj['adj_feature_omics1'].to(self.device)
        self.adj_feature_omics2 = self.adj['adj_feature_omics2'].to(self.device)
        
        # 特征数据
        self.features_omics1 = torch.FloatTensor(self.adata_omics1.obsm['feat'].copy()).to(self.device)
        self.features_omics2 = torch.FloatTensor(self.adata_omics2.obsm['feat'].copy()).to(self.device)
        
        # 维度设置
        self.dim_input1 = self.features_omics1.shape[1]
        self.dim_input2 = self.features_omics2.shape[1]
        self.dim_output1 = self.dim_output
        self.dim_output2 = self.dim_output
        
        # 根据数据类型调整参数
        if self.datatype == 'SPOTS':
            self.epochs = 1  # 只运行一次前向传播
        elif self.datatype == 'Stereo-CITE-seq':
            self.epochs = 1
        elif self.datatype == '10x':
            self.epochs = 1
        elif self.datatype == 'Spatial-epigenome-transcriptome': 
            self.epochs = 1
    
    def generate_representations(self):
        """
        生成四个Graph-specific representation
        
        Returns
        -------
        output : dict
            包含四个Graph-specific representation的字典:
            - emb_latent_spatial_omics1: 模态1的空间图表示
            - emb_latent_spatial_omics2: 模态2的空间图表示  
            - emb_latent_feature_omics1: 模态1的特征图表示
            - emb_latent_feature_omics2: 模态2的特征图表示
        """
        print("Generating Graph-specific representations...")
        
        # 初始化模型
        self.model = Encoder_overall(self.dim_input1, self.dim_output1, self.dim_input2, self.dim_output2).to(self.device)
        
        # 不进行优化，只运行前向传播
        self.model.eval()
        with torch.no_grad():
            # 使用tqdm显示进度，但实际上只运行一次
            for epoch in tqdm(range(self.epochs), desc="Generating representations"):
                results = self.model(self.features_omics1, self.features_omics2, 
                                   self.adj_spatial_omics1, self.adj_feature_omics1, 
                                   self.adj_spatial_omics2, self.adj_feature_omics2)
        
        # 直接返回四个表示，不进行归一化
        output = {
            'emb_latent_spatial_omics1': results['emb_latent_spatial_omics1'].detach().cpu().numpy(),
            'emb_latent_spatial_omics2': results['emb_latent_spatial_omics2'].detach().cpu().numpy(),
            'emb_latent_feature_omics1': results['emb_latent_feature_omics1'].detach().cpu().numpy(),
            'emb_latent_feature_omics2': results['emb_latent_feature_omics2'].detach().cpu().numpy()
        }
        
        print("Graph-specific representations generation finished!")
        print(f"Shape of representations: {output['emb_latent_spatial_omics1'].shape}")
        
        return output

    def get_model_info(self):
        """
        获取模型信息
        
        Returns
        -------
        info : dict
            包含模型配置信息的字典
        """
        info = {
            'datatype': self.datatype,
            'device': str(self.device),
            'random_seed': self.random_seed,
            'dim_input_omics1': self.dim_input1,
            'dim_input_omics2': self.dim_input2,
            'dim_output': self.dim_output,
            'n_cells_omics1': self.adata_omics1.n_obs,
            'n_cells_omics2': self.adata_omics2.n_obs
        }
        return info