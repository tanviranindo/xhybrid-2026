"""
Fusion Module for Multi-Modal Feature Integration
Implements attention-based fusion for syntactic, graph, and temporal features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multiple feature modalities.
    Learns to weight different components adaptively.
    """

    def __init__(self,
                 feature_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            feature_dims: Dictionary mapping modality names to their dimensions
                         e.g., {'syntactic': 128, 'graph': 128, 'temporal': 128}
            hidden_dim: Hidden dimension for projection
            output_dim: Output embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.modality_names = list(feature_dims.keys())
        self.num_modalities = len(feature_dims)

        # Project all modalities to same dimension
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for name, dim in feature_dims.items()
        })

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(
            torch.ones(self.num_modalities) / self.num_modalities
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * self.num_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Store attention weights for explainability
        self.attention_weights = None

    def forward(self,
                features: Dict[str, torch.Tensor],
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Dictionary of feature tensors per modality
            return_attention: Whether to return attention weights

        Returns:
            (batch_size, output_dim) fused embeddings
        """
        batch_size = next(iter(features.values())).size(0)
        device = next(iter(features.values())).device

        # Project all features
        projected = {}
        for name in self.modality_names:
            if name in features:
                projected[name] = self.projections[name](features[name])
            else:
                # If modality missing, use zeros
                projected[name] = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Stack as sequence for attention
        feature_seq = torch.stack(
            [projected[name] for name in self.modality_names],
            dim=1
        )  # (batch, num_modalities, hidden_dim)

        # Self-attention across modalities
        attended, attn_weights = self.cross_attention(
            feature_seq, feature_seq, feature_seq
        )
        self.attention_weights = attn_weights

        # Weight by learned modality importance
        weights = F.softmax(self.modality_weights, dim=0)
        weighted = attended * weights.unsqueeze(0).unsqueeze(-1)

        # Concatenate and project
        fused = weighted.reshape(batch_size, -1)
        output = self.output_proj(fused)

        if return_attention:
            return output, attn_weights
        return output

    def get_modality_importance(self) -> Dict[str, float]:
        """Get normalized importance weights for each modality."""
        weights = F.softmax(self.modality_weights, dim=0)
        return {
            name: weights[i].item()
            for i, name in enumerate(self.modality_names)
        }


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for adaptive feature combination.
    Uses gating to control information flow from each modality.
    """

    def __init__(self,
                 feature_dims: Dict[str, int],
                 output_dim: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            feature_dims: Dictionary mapping modality names to their dimensions
            output_dim: Output embedding dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.modality_names = list(feature_dims.keys())
        total_dim = sum(feature_dims.values())

        # Gate generators for each modality
        self.gates = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(total_dim, dim),
                nn.Sigmoid()
            )
            for name, dim in feature_dims.items()
        })

        # Transform for each modality
        self.transforms = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, output_dim // len(feature_dims)),
                nn.LayerNorm(output_dim // len(feature_dims)),
                nn.ReLU()
            )
            for name, dim in feature_dims.items()
        })

        # Final projection
        adjusted_dim = (output_dim // len(feature_dims)) * len(feature_dims)
        self.output_proj = nn.Sequential(
            nn.Linear(adjusted_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

        self.output_dim = output_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Dictionary of feature tensors per modality

        Returns:
            (batch_size, output_dim) fused embeddings
        """
        # Concatenate all features for gating context
        all_features = torch.cat(
            [features[name] for name in self.modality_names],
            dim=-1
        )

        # Apply gates and transforms
        gated_features = []
        for name in self.modality_names:
            gate = self.gates[name](all_features)
            gated = features[name] * gate
            transformed = self.transforms[name](gated)
            gated_features.append(transformed)

        # Concatenate and project
        fused = torch.cat(gated_features, dim=-1)
        return self.output_proj(fused)


class HybridFusion(nn.Module):
    """
    Hybrid fusion combining attention and gating mechanisms.
    Designed specifically for syntactic-graph-temporal fusion.
    """

    def __init__(self,
                 syntactic_dim: int = 128,
                 graph_dim: int = 128,
                 temporal_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            syntactic_dim: Dimension of syntactic features
            graph_dim: Dimension of graph features
            temporal_dim: Dimension of temporal features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        # Individual modality projections
        self.syntactic_proj = nn.Linear(syntactic_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)
        self.temporal_proj = nn.Linear(temporal_dim, hidden_dim)

        # Pairwise interaction layers
        self.syntactic_graph = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim // 2)
        self.syntactic_temporal = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim // 2)
        self.graph_temporal = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim // 2)

        # Attention-based aggregation
        interaction_dim = hidden_dim * 3 + (hidden_dim // 2) * 3
        self.attention_pool = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Final output projection
        self.output_proj = nn.Sequential(
            nn.Linear(interaction_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        self.output_dim = output_dim

        # Component importance tracking
        self.component_scores = None

    def forward(self,
                syntactic: torch.Tensor,
                graph: torch.Tensor,
                temporal: torch.Tensor,
                return_components: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            syntactic: (batch_size, syntactic_dim) syntactic features
            graph: (batch_size, graph_dim) graph features
            temporal: (batch_size, temporal_dim) temporal features
            return_components: Whether to return component contributions

        Returns:
            (batch_size, output_dim) fused embeddings
        """
        # Project to common dimension
        h_syn = self.syntactic_proj(syntactic)
        h_graph = self.graph_proj(graph)
        h_temp = self.temporal_proj(temporal)

        # Compute pairwise interactions
        syn_graph = F.relu(self.syntactic_graph(h_syn, h_graph))
        syn_temp = F.relu(self.syntactic_temporal(h_syn, h_temp))
        graph_temp = F.relu(self.graph_temporal(h_graph, h_temp))

        # Concatenate all representations
        combined = torch.cat([
            h_syn, h_graph, h_temp,  # Individual
            syn_graph, syn_temp, graph_temp  # Pairwise
        ], dim=-1)

        # Output projection
        output = self.output_proj(combined)

        if return_components:
            # Compute component contributions
            self.component_scores = {
                'syntactic': torch.norm(h_syn, dim=-1),
                'graph': torch.norm(h_graph, dim=-1),
                'temporal': torch.norm(h_temp, dim=-1),
                'syntactic_graph': torch.norm(syn_graph, dim=-1),
                'syntactic_temporal': torch.norm(syn_temp, dim=-1),
                'graph_temporal': torch.norm(graph_temp, dim=-1)
            }
            return output, self.component_scores

        return output


class ResidualFusion(nn.Module):
    """
    Fusion with residual connections for gradient flow.
    """

    def __init__(self,
                 feature_dims: Dict[str, int],
                 output_dim: int = 256,
                 num_blocks: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            feature_dims: Dictionary mapping modality names to dimensions
            output_dim: Output embedding dimension
            num_blocks: Number of residual blocks
            dropout: Dropout rate
        """
        super().__init__()

        self.modality_names = list(feature_dims.keys())
        total_dim = sum(feature_dims.values())

        # Initial projection
        self.input_proj = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            for _ in range(num_blocks)
        ])

        self.output_dim = output_dim

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Concatenate features
        combined = torch.cat(
            [features[name] for name in self.modality_names],
            dim=-1
        )

        # Initial projection
        x = self.input_proj(combined)

        # Residual blocks
        for block in self.res_blocks:
            x = x + block(x)
            x = F.relu(x)

        return x


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns sample-specific weights.
    Different samples may need different modality contributions.
    """

    def __init__(self,
                 feature_dims: Dict[str, int],
                 hidden_dim: int = 128,
                 output_dim: int = 256,
                 dropout: float = 0.1):
        """
        Args:
            feature_dims: Dictionary mapping modality names to dimensions
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.modality_names = list(feature_dims.keys())
        self.num_modalities = len(feature_dims)

        # Projections
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in feature_dims.items()
        })

        # Weight generator (sample-specific)
        total_dim = sum(feature_dims.values())
        self.weight_generator = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_modalities),
            nn.Softmax(dim=-1)
        )

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )

        self.output_dim = output_dim
        self.sample_weights = None

    def forward(self,
                features: Dict[str, torch.Tensor],
                return_weights: bool = False) -> torch.Tensor:
        """
        Forward pass with sample-specific adaptive weights.
        """
        # Concatenate for weight generation
        all_features = torch.cat(
            [features[name] for name in self.modality_names],
            dim=-1
        )

        # Generate sample-specific weights
        weights = self.weight_generator(all_features)  # (batch, num_modalities)
        self.sample_weights = weights

        # Project and weight each modality
        projected = torch.stack([
            self.projections[name](features[name])
            for name in self.modality_names
        ], dim=1)  # (batch, num_modalities, hidden_dim)

        # Weighted combination
        weighted = (projected * weights.unsqueeze(-1)).sum(dim=1)

        output = self.output_proj(weighted)

        if return_weights:
            return output, weights
        return output

    def get_sample_weights(self) -> Optional[torch.Tensor]:
        """Get last computed sample-specific weights."""
        return self.sample_weights
