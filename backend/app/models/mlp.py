import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return x + self.block(x)

class Yoga3HeadMLP(nn.Module):
    def __init__(self, input_dim, num_poses, num_joints=15):
        super(Yoga3HeadMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        self.res1 = ResBlock(256, dropout=0.3)
        self.res2 = ResBlock(256, dropout=0.3)
        
        # Head 1: Pose ID Classification
        self.pose_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_poses)
        )
        
        # Head 2: Correctness Classification (binary probability output)
        self.correctness_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Head 3: Joint Deviation Regression
        self.deviation_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_joints)
        )
        
    def forward(self, x):
        features = self.input_layer(x)
        features = self.res1(features)
        features = self.res2(features)
        
        pose_logits = self.pose_head(features)
        correctness_logit = self.correctness_head(features).squeeze(-1)
        deviations = self.deviation_head(features)
        
        return pose_logits, correctness_logit, deviations
