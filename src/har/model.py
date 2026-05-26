from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = self.conv1(inputs)
        hidden = self.bn1(hidden)
        hidden = self.act(hidden)
        hidden = self.dropout(hidden)
        hidden = self.conv2(hidden)
        hidden = self.bn2(hidden)
        hidden = hidden + residual
        return self.act(hidden)


class SequenceTabularModel(nn.Module):
    def __init__(self, sequence_dim: int, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(sequence_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        self.stem_dropout = nn.Dropout1d(0.1)
        self.encoder = nn.Sequential(
            ResidualConvBlock(64, 128, stride=2, dropout=0.15),
            ResidualConvBlock(128, 128, stride=2, dropout=0.15),
        )
        self.recurrent = nn.GRU(
            input_size=128,
            hidden_size=96,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = nn.Linear(192, 1)
        self.sequence_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),
        )
        self.feature_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, sequences: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        hidden = sequences.transpose(1, 2)
        hidden = self.stem(hidden)
        hidden = self.stem_dropout(hidden)
        hidden = self.encoder(hidden)
        hidden = hidden.transpose(1, 2)
        hidden, _ = self.recurrent(hidden)
        hidden = F.dropout(hidden, p=0.2, training=self.training)

        attn_scores = torch.softmax(self.attention(hidden), dim=1)
        attn_pool = torch.sum(attn_scores * hidden, dim=1)
        max_pool = hidden.max(dim=1).values
        seq_embedding = self.sequence_head(torch.cat([attn_pool, max_pool], dim=1))
        feat_embedding = self.feature_head(features)
        return self.classifier(torch.cat([seq_embedding, feat_embedding], dim=1))
