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
    batch: list, each element is a tuple ([v1, v2, v3, v4], label) returned by Dataset
    return: ([batch_v1, batch_v2, batch_v3, batch_v4], batch_label)
    """
    views_list = []  # store all single samples for each view: views_list[i] = [v_i_0, v_i_1, ..., v_i_batch-1]
    labels = []
    
    for sample in batch:
        x, y = sample  # x is a list of 4 views, y is the label of a single sample
        assert len(x) == 4, f"Error in number of views per sample: expected 4, got {len(x)} (Dataset configuration error)"
        views_list.append(x)  # views_list shape: [batch_size, 4]
        labels.append(y)
    
    # concatenate batch for each view
    batch_views = []
    for view_idx in range(4):  
        # extract all single-sample tensors of the current view and stack into a batch tensor
        view_samples = [sample_views[view_idx] for sample_views in views_list]
        batch_view = torch.stack(view_samples, dim=0)  # stacked as [batch_size, view_dim]
        # key modification: move to GPU directly in collate_fn
        batch_view = batch_view.to(device) 
        batch_views.append(batch_view)
    
    # concatenate labels
    batch_labels = torch.stack(labels, dim=0)
    batch_labels = batch_labels.to(device) 
    
    return batch_views, batch_labels
    
class SMGC:
    
    def __init__(self, mv_dataset, device, lr,  epochs, 
                 latent_dim, p,  batch_size, use_linear_projection, 
                 weight_decay=1e-5, seed=42, loss_weights=[1.0, 1.0],autoencoder_mid_archs=[128,64,32]):
        """        
        Parameters:
        - mv_dataset: multi-view dataset
        - device: training device
        - lr: learning rate
        - weight_decay: weight decay
        - epochs: number of training epochs
        - seed: random seed
        - latent_dim: latent space dimension
        - p: granule parameter
        - loss_weights: loss weights [contrastive loss weight, reconstruction loss weight]
        - batch_size: batch size
        - autoencoder_mid_archs: middle layer architecture of the autoencoder
        - use_linear_projection: whether to use linear projection
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
        

        assert hasattr(self.mv_dataset, 'labels') and self.mv_dataset.labels is not None, \
            "The dataset must contain a 'labels' attribute"
        
        # set random seed
        init_torch(seed=self.seed)
        
        # build data loader
        if self.batch_size == -1:
            self.batch_size = len(self.mv_dataset)
        
        def custom_collate_fn(batch):
            views_list = []
            labels = []
            
            for sample in batch:
                x, y = sample
                assert len(x) == 4, f"Error in number of views per sample: expected 4, got {len(x)}"
                views_list.append(x)
                labels.append(y)
            
            batch_views = []
            for view_idx in range(4):
                view_samples = [sample_views[view_idx] for sample_views in views_list]
                batch_view = torch.stack(view_samples, dim=0)
                batch_view = batch_view.to(self.device)  
                batch_views.append(batch_view)
            
            batch_labels = torch.stack(labels, dim=0)
            batch_labels = batch_labels.to(self.device)
            
            return batch_views, batch_labels
        
        self.dataloader = DataLoader(
            self.mv_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        # 
        self.eval_loader = DataLoader(
            self.mv_dataset,
            batch_size=self.batch_size,
            shuffle=False,        
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        # build model
        self.model = self._build_model()
        self.model.to(self.device)
        
        # build optimizer and loss functions
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
        self.criterion_ins = ContrastiveLoss() 
    
    def _build_model(self):
        """Build the multi-view autoencoder model"""
        # automatically generate the middle layer structure for each view
        middle_encoders = [self.autoencoder_mid_archs[:] for _ in range(self.mv_dataset.num_view)]
        
        # build multi-view autoencoder
        mv_aes = MultiviewAutoEncoder(
            self.mv_dataset.view_dims,
            self.latent_dim,
            middle_encoders,
            self.use_linear_projection
        )
        
        # add a normalization layer after the encoder
        for v in range(self.mv_dataset.num_view):
            mv_aes[v].encoder.middle_layers.append(Normalize())
            
        return mv_aes
    
    def train(self):
        """Train the model and return the final feature representation"""
        # model training process
        for epoch in range(self.epochs):
            loss_con_avg = 0
            loss_rec_avg = 0
            self.model.train()
            
            for bid, batch_data in enumerate(self.dataloader):
                # process data format
                if isinstance(batch_data, tuple) and len(batch_data) == 2:
                    x, y = batch_data
                else:
                    raise ValueError(f"Dataset return format error: expected (tuple([4-view data], label)), got {type(batch_data)}")
                
               
               
                x = [x_v.to(self.device) for x_v in x] 
                current_y = y.to(self.device)  
                
                # forward pass: obtain hidden representations and reconstructed data
                hs, x_rs = self.model(x)
                
                # compute reconstruction loss (sum over 4 views)
                loss_rec = torch.tensor(0., device=self.device)
                for v in range(len(x)):
                    loss_rec += self.criterion_rec(x[v], x_rs[v])
                
                # compute granule-based contrastive loss
                mv_gblist = MVGBList(hs, current_y, self.p)
                loss_con = self.criterion_gra(mv_gblist)
                
                # combine losses and backpropagate
                loss = loss_con * self.loss_weights[0] + loss_rec * self.loss_weights[1]
                loss_con_avg += loss_con.item()
                loss_rec_avg += loss_rec.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # compute average loss and update learning rate
            loss_con_avg /= len(self.dataloader)
            loss_rec_avg /= len(self.dataloader)
            self.scheduler.step()
            
            print(f"epoch {epoch + 1} | loss_con={round(loss_con_avg, 4):.4f}, loss_rec={round(loss_rec_avg, 4):.4f}")

        # obtain final feature representations
        self.model.eval()
        with torch.no_grad():
            # load all data through dataloader and merge
            all_data = []
            for batch in self.eval_loader:
                x_batch, _ = batch  # only take 4-view data, ignore labels
                all_data.append(x_batch)
            
            # merge all batches for each view
            data = []
            for view_idx in range(self.mv_dataset.num_view):
                view_data = torch.cat([batch[view_idx] for batch in all_data])
                data.append(view_data.to(self.device))
            
            # extract and aggregate hidden features of all samples
            hs, _ = self.model(data)
            final_features = torch.stack(hs, dim=0).mean(0).detach().cpu().numpy()
        
        return final_features
