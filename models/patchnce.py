import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSampleF(nn.Module):
    """
    2-layer MLP per feature layer.
    Samples num_patches spatial locations, projects them to a shared
    embed_dim space, and L2-normalizes the result.

    Used to map encoder features (from both input and output) into a
    common embedding space for the InfoNCE contrastive loss.
    """
    def __init__(self, in_channels_list, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(c, embed_dim),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dim, embed_dim),
            )
            for c in in_channels_list
        ])

    def forward(self, feats, num_patches, patch_ids=None):
        """
        feats       : list of [B, C, H, W] tensors, one per feature layer
        num_patches : number of spatial locations to sample
        patch_ids   : if given, reuse the same locations (for matching query/key)
        Returns     : (projections, patch_ids)
                      projections[i] is [B*num_patches, embed_dim], L2-normalized
        """
        results, returned_ids = [], []
        for i, (feat, mlp) in enumerate(zip(feats, self.mlps)):
            B, C, H, W = feat.shape
            n_locs = H * W
            k = min(num_patches, n_locs)
            if patch_ids is not None and i < len(patch_ids):
                ids = patch_ids[i]
            else:
                ids = torch.randperm(n_locs, device=feat.device)[:k]
            feat_flat = feat.permute(0, 2, 3, 1).reshape(B, n_locs, C)
            sampled   = feat_flat[:, ids, :].reshape(B * k, C)
            proj      = F.normalize(mlp(sampled), dim=1)
            results.append(proj)
            returned_ids.append(ids)
        return results, returned_ids


class PatchNCELoss(nn.Module):
    """
    InfoNCE contrastive loss over spatially-sampled feature patches.

    Positive pair: (query[i], key[i]) — same spatial location.
    Negatives:     (query[i], key[j≠i]) — all other sampled locations.
    """
    def __init__(self, tau=0.07):
        super().__init__()
        self.tau = tau
        self.ce  = nn.CrossEntropyLoss()

    def forward(self, feat_q, feat_k):
        """
        feat_q, feat_k : [B*P, D] L2-normalized embeddings
        """
        sim    = torch.mm(feat_q, feat_k.T) / self.tau  # [B*P, B*P]
        labels = torch.arange(sim.size(0), device=feat_q.device)
        return self.ce(sim, labels)
