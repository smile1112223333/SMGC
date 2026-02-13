import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from granular.base import MVGBList
from granular.granular_loss import MultiviewGCLoss
from model.autoencoder import MultiviewAutoEncoder, Normalize
from model.loss import ContrastiveLoss
from utils import init_torch

def custom_collate_fn(batch):
    """
    batch: 列表，每个元素是Dataset返回的 ([v1, v2, v3, v4], label) 元组
    返回：([batch_v1, batch_v2, batch_v3, batch_v4], batch_label)
    """
    views_list = []  # 存储每个视图的所有单样本：views_list[i] = [v_i_0, v_i_1, ..., v_i_batch-1]
    labels = []
    
    for sample in batch:
        x, y = sample  # x是4个视图的list，y是单样本标签
        assert len(x) == 4, f"单样本视图数错误：预期4个，实际{len(x)}个（Dataset配置错误）"
        views_list.append(x)  # views_list形状：[batch_size, 4]
        labels.append(y)
    
    # 手动拼接每个视图的batch（关键：确保4个视图都被拼接）
    batch_views = []
    for view_idx in range(4):  # 强制按4个视图处理（Dataset已确保4个视图存在）
        # 提取当前视图的所有单样本tensor，拼接成batch tensor
        view_samples = [sample_views[view_idx] for sample_views in views_list]
        batch_view = torch.stack(view_samples, dim=0)  # 拼接为 [batch_size, view_dim]
        # 关键修改：在collate_fn中直接移到GPU
        batch_view = batch_view.to(device)  # 添加这一行
        batch_views.append(batch_view)
    
    # 拼接标签
    batch_labels = torch.stack(labels, dim=0)
    batch_labels = batch_labels.to(device)  # 标签也移到GPU
    
    return batch_views, batch_labels
    
class SMGC:
    
    def __init__(self, mv_dataset, device, lr,  epochs, 
                 latent_dim, p,  batch_size, use_linear_projection, 
                 weight_decay=1e-5, seed=42, loss_weights=[1.0, 1.0],autoencoder_mid_archs=[128,64,32]):
        """
        初始化多视图粒球对比聚类训练器（仅使用真实标签）
        
        参数:
        - mv_dataset: 多视图数据集（包含4个视图数据和真实标签）
        - device: 训练设备
        - lr: 学习率
        - weight_decay: 权重衰减
        - epochs: 训练轮数
        - seed: 随机种子
        - latent_dim: 潜在空间维度
        - p: 粒球参数
        - loss_weights: 损失权重 [对比损失权重, 重建损失权重]
        - batch_size: 批次大小
        - autoencoder_mid_archs: 自编码器中间层结构
        - use_linear_projection: 是否使用线性投影
        """
        self.mv_dataset = mv_dataset
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.seed = seed
        self.latent_dim = latent_dim
        self.p = p
        self.loss_weights = loss_weights
        self.batch_size = batch_size
        self.autoencoder_mid_archs = autoencoder_mid_archs
        self.use_linear_projection = use_linear_projection
        
        # 校验数据集是否包含标签
        assert hasattr(self.mv_dataset, 'labels') and self.mv_dataset.labels is not None, \
            "数据集必须包含labels属性（真实标签）"
        
        # 设置随机种子
        init_torch(seed=self.seed)
        
        # 构建数据加载器
        if self.batch_size == -1:
            self.batch_size = len(self.mv_dataset)
        
        # 修改collate_fn，使其能够访问device
        def custom_collate_fn(batch):
            views_list = []
            labels = []
            
            for sample in batch:
                x, y = sample
                assert len(x) == 4, f"单样本视图数错误：预期4个，实际{len(x)}个"
                views_list.append(x)
                labels.append(y)
            
            batch_views = []
            for view_idx in range(4):
                view_samples = [sample_views[view_idx] for sample_views in views_list]
                batch_view = torch.stack(view_samples, dim=0)
                batch_view = batch_view.to(self.device)  # 关键：直接移到设备
                batch_views.append(batch_view)
            
            batch_labels = torch.stack(labels, dim=0)
            batch_labels = batch_labels.to(self.device)  # 标签也移到设备
            
            return batch_views, batch_labels
        
        self.dataloader = DataLoader(
            self.mv_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        # ✅ 新增：评估阶段不 shuffle
        self.eval_loader = DataLoader(
            self.mv_dataset,
            batch_size=self.batch_size,
            shuffle=False,        # ✅ 必须设置为 False
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        # 构建模型
        self.model = self._build_model()
        self.model.to(self.device)
        
        # 构建优化器和损失函数
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=self.epochs, 
            eta_min=0.
        )
        
        self.criterion_rec = nn.MSELoss()
        self.criterion_gra = MultiviewGCLoss()
        self.criterion_ins = ContrastiveLoss() #当p=1时会用到的实例级对比损失这里没有p=1所以代码中就没有用到这一部分
    
    def _build_model(self):
        """构建多视图自编码器模型"""
        # 自动根据视图数生成每个视图的中间层结构
        middle_encoders = [self.autoencoder_mid_archs[:] for _ in range(self.mv_dataset.num_view)]
        
        # 构建多视图自编码器
        mv_aes = MultiviewAutoEncoder(
            self.mv_dataset.view_dims,
            self.latent_dim,
            middle_encoders,
            self.use_linear_projection
        )
        
        # 在编码层后，加一层标准化层
        for v in range(self.mv_dataset.num_view):
            mv_aes[v].encoder.middle_layers.append(Normalize())
            
        return mv_aes
    
    def train(self):
        """训练模型并返回最终特征表示（仅使用真实标签）"""
        # 模型训练过程
        for epoch in range(self.epochs):
            loss_con_avg = 0
            loss_rec_avg = 0
            self.model.train()
            
            for bid, batch_data in enumerate(self.dataloader):
                # 处理数据格式（Dataset返回的是(x, y)元组，x为4视图数据，y为真实标签）
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    x, y = batch_data
                else:
                    raise ValueError(f"数据集返回格式错误：预期(tuple([4视图数据], 标签))，实际{type(batch_data)}")
                
                # # 调试打印（可选：验证数据格式，首次运行建议保留）
                # if bid == 0 and epoch == 0:
                #     print(f"Epoch 0 Batch 0 数据格式验证：")
                #     print(f"- x类型: {type(x)}, 视图数: {len(x)}")
                #     for i, view_batch in enumerate(x):
                #         print(f"  视图{i+1}: 类型={type(view_batch)}, 形状={view_batch.shape}, 设备={view_batch.device}")
                #     print(f"- 标签类型: {type(y)}, 形状={y.shape}, 设备={y.device}")
                    
                # 将数据和标签移动到设备
                x = [x_v.to(self.device) for x_v in x]  # 4个视图数据移设备
                current_y = y.to(self.device)  # 真实标签移设备
                
                # 前向传播：获取隐藏表示和重建数据
                hs, x_rs = self.model(x)
                
                # 计算重建损失（4个视图的重建损失求和）
                loss_rec = torch.tensor(0., device=self.device)
                for v in range(len(x)):
                    loss_rec += self.criterion_rec(x[v], x_rs[v])
                
                # 计算粒球对比损失（使用真实标签）
                mv_gblist = MVGBList(hs, current_y, self.p)
                loss_con = self.criterion_gra(mv_gblist)
                
                # 组合损失并反向传播
                loss = loss_con * self.loss_weights[0] + loss_rec * self.loss_weights[1]
                loss_con_avg += loss_con.item()
                loss_rec_avg += loss_rec.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # 计算平均损失并更新学习率
            loss_con_avg /= len(self.dataloader)
            loss_rec_avg /= len(self.dataloader)
            self.scheduler.step()
            
            print(f"epoch {epoch + 1} | loss_con={round(loss_con_avg, 4):.4f}, loss_rec={round(loss_rec_avg, 4):.4f}")

        # 获取最终的特征表示（所有样本的聚合特征）
        self.model.eval()
        with torch.no_grad():
            # 通过数据加载器获取所有数据并合并
            all_data = []
            for batch in self.eval_loader:
                x_batch, _ = batch  # 只取4视图数据，忽略标签
                all_data.append(x_batch)
            
            # 合并每个视图的所有批次数据
            data = []
            for view_idx in range(self.mv_dataset.num_view):
                view_data = torch.cat([batch[view_idx] for batch in all_data])
                data.append(view_data.to(self.device))
            
            # 提取所有样本的隐藏特征并聚合
            hs, _ = self.model(data)
            final_features = torch.stack(hs, dim=0).mean(0).detach().cpu().numpy()
        
        return final_features