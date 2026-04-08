import torch
import numpy as np
from granular.base import GranularBall, GBList, MVGBList
from granular.tools import relation_of_views_gblists, merge_tensors, relation_of_views_gblists_tensor


class GranularContrastiveLoss(torch.nn.Module):
    # Contrastive learning: bring neighboring balls closer, push non-neighboring balls apart
    # The affinity matrix is used to identify positive and negative samples
    def __init__(self, temperature=1.):
        super(GranularContrastiveLoss, self).__init__()
        self.t = temperature

    def forward(self, gblist):
        pos_mask = gblist.affinity()
        neg_mask = 1 - pos_mask
        num_ins = len(gblist)
        idx = torch.arange(0, num_ins)
        # Correct the positive sample pair mask
        pos_mask[idx, idx] = 0
        x = gblist.get_centers()
        # Compute similarity, here using matrix multiplication
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
        sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
        # Consider rewriting with cross entropy
        sim_pos = pos_mask * sim_x / self.t
        sim_neg = neg_mask * sim_x / self.t
        exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
        expsum_sim = torch.exp(sim_pos) + exp_sim_neg
        # expsum_sim = exp_sim_neg
        loss = -(sim_pos - torch.log(expsum_sim) * pos_mask)

        avg_sim_pos = torch.sum(sim_pos) / torch.sum(pos_mask)
        avg_sim_neg = torch.sum(sim_neg) / (torch.sum(neg_mask))
        return torch.sum(torch.as_tensor(loss)) / num_ins, avg_sim_pos, avg_sim_neg


class MultiviewGCLoss(torch.nn.Module):
    def __init__(self, temperature=1.):
        super(MultiviewGCLoss, self).__init__()
        self.t = temperature

    def forward(self, views: MVGBList):
        # unify device
        device = views[0].data.device
        loss = torch.tensor(0., device=device)
        # perform contrastive learning between every pair of views
        num_views = len(views)
        for i in range(num_views):
            # mask_i_intra = views[i].affinity()
            mask_i_intra = torch.eye(len(views[i]), device=device)
            for j in range(i + 1, num_views):
                # compute masks
                # mask_j_intra = views[j].affinity()
                mask_j_intra = torch.eye(len(views[j]), device=device)
                mask_inter = relation_of_views_gblists_tensor(views[i], views[j])
                # number of granular balls in two views
                ni, nj = len(views[i]), len(views[j])
                # merge intra-view and inter-view mask matrices
                pos_mask = merge_tensors(ni, nj, mask_i_intra, mask_inter, mask_inter.T, mask_j_intra).to(device)
                # neg_mask = 1 - pos_mask
                neg_mask = torch.ones_like(pos_mask).to(device) - pos_mask
                num_ins = ni + nj
                # idx = torch.arange(0, num_ins)
                # correct positive sample pair mask
                # pos_mask[idx, idx] = 0
                centers_i = views[i].get_centers()
                centers_j = views[j].get_centers()
                x = torch.concat((centers_i, centers_j), dim=0)
                # compute similarity, here using matrix multiplication
                norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
                sim_x = x @ x.T / (norm_x @ norm_x.T + 1e-12)
                # consider rewriting with cross entropy
                sim_pos = pos_mask * sim_x / self.t
                sim_neg = neg_mask * sim_x / self.t
                exp_sim_neg = torch.sum(torch.exp(sim_neg), dim=1, keepdim=True).expand((num_ins, num_ins))
                expsum_sim = torch.exp(sim_pos) + exp_sim_neg
                # expsum_sim = exp_sim_neg
                loss += torch.sum(-(sim_pos - torch.log(expsum_sim) * pos_mask)) / pos_mask.sum()
        return loss / (num_views * (num_views - 1) / 2)
