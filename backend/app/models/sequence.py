import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# MediaPipe skeletal anatomical connections
MEDIAPIPE_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (3, 7), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (17, 19),
    (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

def get_normalized_adjacency():
    """
    Builds the normalized skeletal adjacency matrix A_norm for the 33 MediaPipe joints.
    A_norm = D^(-1/2) * (A + I) * D^(-1/2)
    """
    num_nodes = 33
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # Add skeletal edges
    for i, j in MEDIAPIPE_EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
        
    # Add self loops
    A_loop = A + np.eye(num_nodes, dtype=np.float32)
    
    # Calculate degree matrix D
    deg = np.sum(A_loop, axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg>0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    
    # Symmetric normalization
    A_norm = D_inv_sqrt.dot(A_loop).dot(D_inv_sqrt)
    return torch.tensor(A_norm, dtype=torch.float32)


class SpatialGraphConv(nn.Module):
    """
    True Spatial Graph Convolutional Layer.
    Aggregates joint features along the normalized skeletal adjacency matrix A.
    """
    def __init__(self, in_channels, out_channels, A_norm):
        super(SpatialGraphConv, self).__init__()
        self.register_buffer('A', A_norm)  # Keep adjacency matrix on the correct device
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, num_nodes, seq_len] -> [N, C, 33, T]
        # 1. Project features: W * X
        x = self.conv(x)  # [N, C_out, 33, T]
        
        # 2. Graph aggregation: A * (W * X)
        # We multiply adjacency matrix A (shape [33, 33]) across the node dimension (dim=2)
        out = torch.einsum('vw,ncwt->ncvt', self.A, x)
        return out


class TemporalConv(nn.Module):
    """
    Temporal Convolutional Layer.
    Convolves features along the temporal dimension for each node independently.
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.3):
        super(TemporalConv, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, in_channels, num_nodes, seq_len] -> [N, C, 33, T]
        # Swap node and temporal dimensions to convolve over time: [N, C, T, 33]
        x = x.transpose(2, 3)
        x = self.conv(x)
        x = self.bn(x)
        x = self.dropout(x)
        # Swap back: [N, C, 33, T]
        return x.transpose(2, 3)


class STGCNBlock(nn.Module):
    """
    A single Spatio-Temporal Graph Convolutional Block containing:
    1. Spatial Graph Convolution (GCN)
    2. Temporal Graph Convolution (TCN)
    """
    def __init__(self, in_channels, out_channels, A_norm, stride=1, dropout=0.3):
        super(STGCNBlock, self).__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A_norm)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=9, stride=stride, dropout=dropout)
        
        # Residual skip connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
            
    def forward(self, x):
        # x shape: [N, C_in, 33, T]
        res = self.residual(x)
        x = self.gcn(x)
        x = F.gelu(x)
        x = self.tcn(x)
        return F.gelu(x + res)


# True ST-GCN Sequence Classifier (maintaining class name YogaSequenceLSTM for backend loading compatibility)
class YogaSequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(YogaSequenceLSTM, self).__init__()
        A_norm = get_normalized_adjacency()
        
        # Input channels: 3 (x, y, z joint coordinates)
        self.block1 = STGCNBlock(3, 64, A_norm, stride=1, dropout=0.2)
        self.block2 = STGCNBlock(64, 128, A_norm, stride=1, dropout=0.3)
        self.block3 = STGCNBlock(128, 256, A_norm, stride=1, dropout=0.3)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Input x shape: [batch_size, seq_len, input_dim] -> [batch_size, 60, 99]
        # Reshape to [batch_size, seq_len, 33, 3] and permute to [batch_size, channels=3, nodes=33, seq_len=60]
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, 60, 33, 3).permute(0, 3, 2, 1) # [batch_size, 3, 33, 60]
        
        # ST-GCN convolutions
        out = self.block1(x_reshaped)  # [batch_size, 64, 33, 60]
        out = self.block2(out)         # [batch_size, 128, 33, 60]
        out = self.block3(out)         # [batch_size, 256, 33, 60]
        
        # Global Pooling over both Spatial nodes (33) and Temporal frames (60)
        # out shape: [batch_size, 256]
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(out.size(0), -1)
        
        return self.fc(out)
