import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Source: https://github.com/adambielski/siamese-triplet/blob/master/losses.py
"""


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-20

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin, swap=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.swap = swap

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        if self.swap:
            distance_positive_negative = (positive - negative).pow(2).sum(1)
            distance_negative[distance_positive_negative < distance_negative] = distance_positive_negative[distance_positive_negative < distance_negative]
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
class TrajectoryLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        """
        eps: small constant to avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, y_traj: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature-based loss for an n-dimensional trajectory.

        Parameters:
          y_traj: Tensor of shape [T, D], sequence of points.

        Returns:
          loss: scalar tensor, sum of curvature at each segment.
        """
        # 1) form triples: y0, y1, y2 each [T-2, D]
        y0 = y_traj[:-2]
        y1 = y_traj[1:-1]
        y2 = y_traj[2:]

        # 2) finite differences
        v = y1 - y0                   # approximate velocity * dt, shape [T-2, D]
        a = y2 - 2*y1 + y0            # approximate acceleration * dt^2, shape [T-2, D]

        # 3) norms and dot products
        v_norm_sq = v.pow(2).sum(dim=1) + self.eps  # ||v||^2 + eps, shape [T-2]
        a_norm_sq = a.pow(2).sum(dim=1)             # ||a||^2, shape [T-2]
        dot_va    = (v * a).sum(dim=1)              # <v, a>,   shape [T-2]

        # 4) squared numerator of curvature
        #    v^2 * a^2 - <v,a>^2
        cross_sq = v_norm_sq * a_norm_sq - dot_va.pow(2)

        # 5) curvature: sqrt(max(cross_sq, 0)) / (||v||^3)
        curvature = torch.sqrt(torch.clamp(cross_sq, min=0.0)) / (v_norm_sq.pow(1.5) + 1e-8)

        # 6) aggregate loss
        loss = curvature.mean()
        return loss

