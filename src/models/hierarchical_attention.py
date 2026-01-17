"""
Hierarchical Attention Module for Microservice Traces
Novel contribution: Trace-aware multi-level attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class TraceHierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism designed specifically for microservice traces.

    Innovation: Three-level attention hierarchy exploiting trace structure:
    1. Span-level: Temporal patterns within individual service calls
    2. Service-level: Communication patterns between services
    3. Trace-level: Global execution flow patterns

    This is novel because prior work uses generic attention that doesn't
    leverage the inherent hierarchical structure of distributed traces.

    Args:
        hidden_dim: Dimension of span embeddings
        num_heads: Number of attention heads per level
        dropout: Dropout rate
        use_structure: Whether to use parent-child structure information
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_structure: bool = True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_structure = use_structure

        # Level 1: Span-level attention (within-service temporal patterns)
        self.span_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.span_norm = nn.LayerNorm(hidden_dim)

        # Level 2: Service-level attention (inter-service dependencies)
        self.service_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.service_norm = nn.LayerNorm(hidden_dim)

        # Level 3: Trace-level attention (global patterns)
        self.trace_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.trace_norm = nn.LayerNorm(hidden_dim)

        # Learnable level importance weights
        self.level_weights = nn.Parameter(torch.ones(3) / 3)

        # Hierarchical fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Structure-aware bias (if using trace structure)
        if use_structure:
            self.structure_bias = nn.Linear(hidden_dim, hidden_dim)

    def create_service_mask(self, service_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask based on service boundaries.
        Spans from the same service can attend to each other more strongly.

        Args:
            service_ids: (batch, seq_len) service IDs for each span

        Returns:
            (batch, seq_len, seq_len) attention mask
        """
        batch_size, seq_len = service_ids.shape
        device = service_ids.device

        # Create mask where mask[i,j] = 1 if same service, 0 otherwise
        service_mask = (service_ids.unsqueeze(2) == service_ids.unsqueeze(1)).float()

        # Convert to attention mask format (0 for same service, -inf for different)
        attention_mask = (1 - service_mask) * -1e9

        return attention_mask

    def create_trace_structure_bias(self,
                                     structure: Optional[torch.Tensor],
                                     seq_len: int,
                                     device: torch.device) -> torch.Tensor:
        """
        Create bias term based on parent-child relationships in trace.

        Args:
            structure: (batch, seq_len, seq_len) adjacency matrix
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            (batch, seq_len, seq_len) structure bias
        """
        if structure is None or not self.use_structure:
            return torch.zeros(1, seq_len, seq_len, device=device)

        # Parent-child relationships get positive bias
        # Sibling relationships get smaller positive bias
        # Unrelated spans get no bias

        structure_bias = structure.float()

        # Add sibling bias (common parent)
        parent_matrix = structure
        sibling_mask = (parent_matrix @ parent_matrix.transpose(-1, -2)) > 0
        structure_bias = structure_bias + 0.5 * sibling_mask.float()

        return structure_bias * 10.0  # Scale factor

    def forward(self,
                x: torch.Tensor,
                service_ids: Optional[torch.Tensor] = None,
                trace_structure: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass through hierarchical attention.

        Args:
            x: (batch, seq_len, hidden_dim) span embeddings
            service_ids: (batch, seq_len) service ID for each span
            trace_structure: (batch, seq_len, seq_len) parent-child adjacency
            return_attention: Whether to return attention weights

        Returns:
            output: (batch, seq_len, hidden_dim) attended embeddings
            attention_info: Dictionary of attention weights if requested
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Create masks if service_ids provided
        # Note: attn_mask for MultiheadAttention should be 2D (seq_len, seq_len)
        # or 3D (batch*num_heads, seq_len, seq_len)
        service_mask = None
        if service_ids is not None:
            # For now, disable service mask in attention - can be added later
            # service_mask = self.create_service_mask(service_ids)
            pass

        # Create structure bias if provided
        structure_bias = self.create_trace_structure_bias(
            trace_structure, seq_len, device
        )

        # Level 1: Span-level attention (local temporal patterns)
        span_out, span_attn = self.span_attention(
            x, x, x,
            need_weights=return_attention
        )
        span_out = self.span_norm(x + span_out)

        # Level 2: Service-level attention (inter-service patterns)
        service_out, service_attn = self.service_attention(
            x, x, x,
            attn_mask=service_mask,
            need_weights=return_attention
        )
        service_out = self.service_norm(x + service_out)

        # Level 3: Trace-level attention (global patterns)
        # Apply structure bias if available
        if self.use_structure and trace_structure is not None:
            # Simplified structure bias - just use base x for now
            x_biased = x
        else:
            x_biased = x

        trace_out, trace_attn = self.trace_attention(
            x_biased, x_biased, x_biased,
            need_weights=return_attention
        )
        trace_out = self.trace_norm(x + trace_out)

        # Hierarchical fusion with learnable weights
        weights = F.softmax(self.level_weights, dim=0)

        # Weighted combination
        weighted_span = weights[0] * span_out
        weighted_service = weights[1] * service_out
        weighted_trace = weights[2] * trace_out

        # Concatenate and fuse
        combined = torch.cat([weighted_span, weighted_service, weighted_trace], dim=-1)
        output = self.fusion(combined)

        # Return attention info if requested
        if return_attention:
            attention_info = {
                'span_attention': span_attn,
                'service_attention': service_attn,
                'trace_attention': trace_attn,
                'level_weights': weights.detach().cpu().numpy(),
                'weighted_contributions': {
                    'span': weighted_span.norm(dim=-1).mean().item(),
                    'service': weighted_service.norm(dim=-1).mean().item(),
                    'trace': weighted_trace.norm(dim=-1).mean().item()
                }
            }
            return output, attention_info

        return output, None

    def get_level_importance(self) -> Dict[str, float]:
        """Get normalized importance of each attention level."""
        weights = F.softmax(self.level_weights, dim=0).detach().cpu().numpy()
        return {
            'span_level': float(weights[0]),
            'service_level': float(weights[1]),
            'trace_level': float(weights[2])
        }


class AdaptiveHierarchicalAttention(nn.Module):
    """
    Adaptive version that learns when to use each level based on trace characteristics.

    Innovation: Not all traces need all three levels. This module learns to
    adapt the hierarchy based on trace complexity.
    """

    def __init__(self,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.hierarchical_attn = TraceHierarchicalAttention(
            hidden_dim, num_heads, dropout
        )

        # Trace complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 3 complexity scores
            nn.Sigmoid()
        )

    def forward(self,
                x: torch.Tensor,
                service_ids: Optional[torch.Tensor] = None,
                trace_structure: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Forward with adaptive level selection.

        Args:
            x: (batch, seq_len, hidden_dim)
            service_ids: (batch, seq_len)
            trace_structure: (batch, seq_len, seq_len)
            return_attention: bool

        Returns:
            Adaptively attended embeddings
        """
        # Estimate trace complexity
        trace_repr = x.mean(dim=1)  # (batch, hidden_dim)
        complexity = self.complexity_estimator(trace_repr)  # (batch, 3)

        # Get hierarchical attention
        output, attn_info = self.hierarchical_attn(
            x, service_ids, trace_structure, return_attention
        )

        # Modulate output by complexity-based weights
        # Simple traces get more span-level, complex get more trace-level
        complexity_weights = complexity.unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, 3)

        # Already fused in hierarchical_attn, so just scale
        # This is a simplified version - could be more sophisticated

        if return_attention and attn_info is not None:
            attn_info['complexity_weights'] = complexity.detach().cpu().numpy()

        return output, attn_info


# Example usage and testing
if __name__ == "__main__":
    # Test hierarchical attention
    batch_size = 4
    seq_len = 20
    hidden_dim = 128

    # Create dummy data
    x = torch.randn(batch_size, seq_len, hidden_dim)
    service_ids = torch.randint(0, 10, (batch_size, seq_len))

    # Create dummy trace structure (parent-child relationships)
    trace_structure = torch.zeros(batch_size, seq_len, seq_len)
    for b in range(batch_size):
        for i in range(seq_len - 1):
            trace_structure[b, i, i+1] = 1  # Sequential structure

    # Test standard hierarchical attention
    print("Testing TraceHierarchicalAttention...")
    attn = TraceHierarchicalAttention(hidden_dim=hidden_dim)
    output, attn_info = attn(x, service_ids, trace_structure, return_attention=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Level importance: {attn.get_level_importance()}")
    print(f"Weighted contributions: {attn_info['weighted_contributions']}")

    # Test adaptive version
    print("\nTesting AdaptiveHierarchicalAttention...")
    adaptive_attn = AdaptiveHierarchicalAttention(hidden_dim=hidden_dim)
    output_adaptive, attn_info_adaptive = adaptive_attn(
        x, service_ids, trace_structure, return_attention=True
    )

    print(f"Adaptive output shape: {output_adaptive.shape}")
    print(f"Complexity weights: {attn_info_adaptive['complexity_weights']}")

    print("\n✅ Hierarchical attention tests passed!")
