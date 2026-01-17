"""
Temporal Module for Sequence Modeling
Implements BiLSTM and Transformer encoders for service call sequences.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for service call sequences.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 output_dim: int = 128,
                 dropout: float = 0.1,
                 use_latency: bool = True):
        """
        Args:
            vocab_size: Size of service vocabulary
            embedding_dim: Dimension of service embeddings
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output embedding dimension
            dropout: Dropout rate
            use_latency: Whether to use latency features
        """
        super().__init__()

        self.use_latency = use_latency
        self.hidden_dim = hidden_dim

        # Service embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Latency projection
        if use_latency:
            self.latency_proj = nn.Linear(1, embedding_dim // 4)
            lstm_input_dim = embedding_dim + embedding_dim // 4
        else:
            lstm_input_dim = embedding_dim

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Attention for sequence aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.output_dim = output_dim

    def forward(self,
                sequences: torch.Tensor,
                latencies: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: (batch_size, seq_len) service indices
            latencies: (batch_size, seq_len) latency values
            return_attention: Whether to return attention weights

        Returns:
            (batch_size, output_dim) sequence embeddings
        """
        # Create padding mask
        padding_mask = sequences == 0  # (batch, seq_len)

        # Embed sequences
        embedded = self.embedding(sequences)  # (batch, seq_len, embed_dim)

        # Add latency features
        if self.use_latency and latencies is not None:
            lat_embedded = self.latency_proj(latencies.unsqueeze(-1))
            embedded = torch.cat([embedded, lat_embedded], dim=-1)

        # BiLSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # (batch, seq_len, hidden*2)

        # Attention-weighted aggregation
        attn_scores = self.attention(lstm_out).squeeze(-1)  # (batch, seq_len)
        attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)

        # Weighted sum
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden*2)

        # Output projection
        output = self.output_proj(context)

        if return_attention:
            return output, attn_weights
        return output

    def get_sequence_representations(self,
                                      sequences: torch.Tensor,
                                      latencies: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get per-position representations (before aggregation)."""
        embedded = self.embedding(sequences)

        if self.use_latency and latencies is not None:
            lat_embedded = self.latency_proj(latencies.unsqueeze(-1))
            embedded = torch.cat([embedded, lat_embedded], dim=-1)

        lstm_out, _ = self.lstm(embedded)
        return lstm_out


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for service call sequences.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 64,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 ff_dim: int = 256,
                 output_dim: int = 128,
                 max_seq_len: int = 100,
                 dropout: float = 0.1,
                 use_latency: bool = True):
        """
        Args:
            vocab_size: Size of service vocabulary
            embedding_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            output_dim: Output embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout rate
            use_latency: Whether to use latency features
        """
        super().__init__()

        self.use_latency = use_latency
        self.embedding_dim = embedding_dim

        # Service embedding
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        # Latency embedding
        if use_latency:
            self.latency_proj = nn.Linear(1, embedding_dim // 4)
            transformer_dim = embedding_dim + embedding_dim // 4
        else:
            transformer_dim = embedding_dim

        # Project to model dimension
        self.input_proj = nn.Linear(transformer_dim, embedding_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_dim = output_dim

        # Store attention weights
        self.attention_weights = None

    def forward(self,
                sequences: torch.Tensor,
                latencies: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: (batch_size, seq_len) service indices
            latencies: (batch_size, seq_len) latency values
            return_attention: Whether to return attention weights

        Returns:
            (batch_size, output_dim) sequence embeddings
        """
        batch_size = sequences.size(0)

        # Create padding mask for transformer
        padding_mask = sequences == 0  # (batch, seq_len)

        # Embed sequences
        embedded = self.embedding(sequences)  # (batch, seq_len, embed_dim)

        # Add latency features
        if self.use_latency and latencies is not None:
            lat_embedded = self.latency_proj(latencies.unsqueeze(-1))
            embedded = torch.cat([embedded, lat_embedded], dim=-1)

        # Project to model dimension
        embedded = self.input_proj(embedded)

        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat([cls_tokens, embedded], dim=1)

        # Update padding mask for CLS token
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=sequences.device)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        # Add positional encoding
        embedded = self.pos_encoder(embedded)

        # Transformer encoding
        transformer_out = self.transformer(
            embedded,
            src_key_padding_mask=padding_mask
        )

        # Use CLS token representation
        cls_output = transformer_out[:, 0, :]

        # Output projection
        output = self.output_proj(cls_output)

        if return_attention:
            # Get attention from last layer (approximate)
            return output, None  # Full attention extraction requires hooks
        return output

    def get_token_representations(self,
                                   sequences: torch.Tensor,
                                   latencies: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get per-token representations."""
        batch_size = sequences.size(0)
        padding_mask = sequences == 0

        embedded = self.embedding(sequences)
        if self.use_latency and latencies is not None:
            lat_embedded = self.latency_proj(latencies.unsqueeze(-1))
            embedded = torch.cat([embedded, lat_embedded], dim=-1)

        embedded = self.input_proj(embedded)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedded = torch.cat([cls_tokens, embedded], dim=1)

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=sequences.device)
        padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        embedded = self.pos_encoder(embedded)
        return self.transformer(embedded, src_key_padding_mask=padding_mask)


class TemporalEncoder(nn.Module):
    """
    Unified temporal encoder wrapper.
    Supports both LSTM and Transformer architectures.
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 64,
                 hidden_dim: int = 128,
                 output_dim: int = 128,
                 architecture: str = 'lstm',
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_latency: bool = True,
                 max_seq_len: int = 100):
        """
        Args:
            vocab_size: Size of service vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Hidden layer dimension (LSTM) or FF dimension (Transformer)
            output_dim: Output embedding dimension
            architecture: 'lstm' or 'transformer'
            num_layers: Number of encoder layers
            num_heads: Number of attention heads (Transformer only)
            dropout: Dropout rate
            use_latency: Whether to use latency features
            max_seq_len: Maximum sequence length (Transformer only)
        """
        super().__init__()

        self.architecture = architecture

        if architecture == 'lstm':
            self.encoder = BiLSTMEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout,
                use_latency=use_latency
            )
        elif architecture == 'transformer':
            self.encoder = TransformerEncoder(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                ff_dim=hidden_dim * 2,
                output_dim=output_dim,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_latency=use_latency
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.output_dim = output_dim

    def forward(self,
                sequences: torch.Tensor,
                latencies: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """Forward pass through selected encoder."""
        return self.encoder(sequences, latencies, return_attention)

    def get_representations(self,
                            sequences: torch.Tensor,
                            latencies: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get per-position representations."""
        if self.architecture == 'lstm':
            return self.encoder.get_sequence_representations(sequences, latencies)
        else:
            return self.encoder.get_token_representations(sequences, latencies)


class LatencyAnomalyDetector(nn.Module):
    """
    Specialized module for detecting latency-based anomalies.
    Uses 1D-CNN + attention for pattern detection.
    """

    def __init__(self,
                 seq_len: int = 50,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 kernel_sizes: Tuple[int, ...] = (3, 5, 7),
                 dropout: float = 0.1):
        """
        Args:
            seq_len: Input sequence length
            hidden_dim: Hidden dimension per kernel
            output_dim: Output dimension
            kernel_sizes: Tuple of 1D convolution kernel sizes
            dropout: Dropout rate
        """
        super().__init__()

        # Multi-scale 1D convolutions
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_dim, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])

        # Attention aggregation
        conv_out_dim = hidden_dim * len(kernel_sizes)
        self.attention = nn.Sequential(
            nn.Linear(conv_out_dim, conv_out_dim // 2),
            nn.Tanh(),
            nn.Linear(conv_out_dim // 2, 1)
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(conv_out_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

        self.output_dim = output_dim

    def forward(self, latencies: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            latencies: (batch_size, seq_len) latency sequences

        Returns:
            (batch_size, output_dim) latency embeddings
        """
        # Add channel dimension
        x = latencies.unsqueeze(1)  # (batch, 1, seq_len)

        # Apply multi-scale convolutions
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)  # (batch, hidden_dim, seq_len)
            conv_outputs.append(out)

        # Concatenate
        x = torch.cat(conv_outputs, dim=1)  # (batch, hidden_dim * n_kernels, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, features)

        # Attention-weighted aggregation
        attn_scores = self.attention(x).squeeze(-1)  # (batch, seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)

        context = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        return self.output_proj(context)
