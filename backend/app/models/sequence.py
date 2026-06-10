import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        scores = self.attn(x)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights, dim=1)
        return context

class ResGRUBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.3):
        super(ResGRUBlock, self).__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.proj(out)
        out = self.ln(out)
        out = self.dropout(out)
        return x + out

class YogaSequenceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(YogaSequenceLSTM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.res_gru1 = ResGRUBlock(hidden_dim, dropout=0.3)
        self.res_gru2 = ResGRUBlock(hidden_dim, dropout=0.3)
        self.attention = AttentionBlock(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.res_gru1(x)
        x = self.res_gru2(x)
        x = self.attention(x)
        return self.fc(x)
