"""
Graph Neural Network Module for Service Dependency Graphs
Implements GAT and GCN encoders for microservice topology learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import Data, Batch

try:
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data, Batch
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    # Create dummy classes for type hints when torch_geometric is not available
    if not TYPE_CHECKING:
        class Data:
            pass
        class Batch:
            pass


class ServiceGAT(nn.Module):
    """
    Graph Attention Network for service dependency graphs.
    Uses multi-head attention to learn service relationships.
    """

    def __init__(self,
                 in_channels: int = 6,
                 hidden_channels: int = 64,
                 out_channels: int = 128,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 edge_dim: Optional[int] = 1):
        """
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout rate
            edge_dim: Edge feature dimension (None to ignore)
        """
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for ServiceGAT")

        self.num_layers = num_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_channels if i == 0 else hidden_channels * num_heads
            out_dim = hidden_channels if i < num_layers - 1 else out_channels // num_heads

            self.gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=num_heads,
                    concat=True if i < num_layers - 1 else True,
                    dropout=dropout,
                    edge_dim=edge_dim
                )
            )

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels * num_heads if i < num_layers - 1 else out_channels)
            for i in range(num_layers)
        ])

        # Final output dimension
        self.output_dim = out_channels

        # Store attention weights for explainability
        self.attention_weights = None

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass through GAT layers.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            edge_attr: Edge features (num_edges, edge_dim)
            batch: Batch assignment for nodes
            return_attention: Whether to store attention weights

        Returns:
            Graph-level embedding (batch_size, out_channels)
        """
        # Input projection
        h = self.input_proj(x)

        # GAT layers
        for i, gat in enumerate(self.gat_layers):
            if return_attention and i == self.num_layers - 1:
                h, attn = gat(h, edge_index, edge_attr=edge_attr,
                             return_attention_weights=True)
                self.attention_weights = attn
            else:
                h = gat(h, edge_index, edge_attr=edge_attr)

            h = self.layer_norms[i](h)
            if i < self.num_layers - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling
        if batch is not None:
            # Combine mean and max pooling
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h = (h_mean + h_max) / 2
        else:
            # Single graph
            h_mean = h.mean(dim=0, keepdim=True)
            h_max = h.max(dim=0, keepdim=True)[0]
            h = (h_mean + h_max) / 2

        return h

    def get_attention_weights(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return stored attention weights for explainability."""
        return self.attention_weights


class ServiceGCN(nn.Module):
    """
    Graph Convolutional Network for service dependency graphs.
    Simpler alternative to GAT.
    """

    def __init__(self,
                 in_channels: int = 6,
                 hidden_channels: int = 64,
                 out_channels: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super().__init__()

        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric is required for ServiceGCN")

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_channels if i == 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels

            self.gcn_layers.append(
                GCNConv(in_dim, out_dim)
            )

        # Batch norms
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels if i < num_layers - 1 else out_channels)
            for i in range(num_layers)
        ])

        self.output_dim = out_channels

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GCN layers.
        """
        h = x

        for i, gcn in enumerate(self.gcn_layers):
            h = gcn(h, edge_index)
            h = self.batch_norms[i](h)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Global pooling
        if batch is not None:
            h = global_mean_pool(h, batch)
        else:
            h = h.mean(dim=0, keepdim=True)

        return h


class GraphEncoder(nn.Module):
    """
    Wrapper for graph encoding with multiple options.
    Provides unified interface for different GNN architectures.
    """

    def __init__(self,
                 in_channels: int = 6,
                 hidden_channels: int = 64,
                 out_channels: int = 128,
                 architecture: str = 'gat',
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            architecture: 'gat' or 'gcn'
            num_heads: Number of attention heads (GAT only)
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()

        if architecture == 'gat':
            self.encoder = ServiceGAT(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout
            )
        elif architecture == 'gcn':
            self.encoder = ServiceGCN(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.architecture = architecture
        self.output_dim = out_channels

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """Forward pass through the selected encoder."""
        if self.architecture == 'gat':
            return self.encoder(x, edge_index, edge_attr, batch, return_attention)
        else:
            return self.encoder(x, edge_index, edge_attr, batch)

    def get_node_embeddings(self,
                            x: torch.Tensor,
                            edge_index: torch.Tensor,
                            edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get node-level embeddings (before pooling).
        Useful for node-level explainability.
        """
        if self.architecture == 'gat':
            h = self.encoder.input_proj(x)
            for i, gat in enumerate(self.encoder.gat_layers):
                h = gat(h, edge_index, edge_attr=edge_attr)
                h = self.encoder.layer_norms[i](h)
                if i < self.encoder.num_layers - 1:
                    h = F.elu(h)
        else:
            h = x
            for i, gcn in enumerate(self.encoder.gcn_layers):
                h = gcn(h, edge_index)
                h = self.encoder.batch_norms[i](h)
                if i < self.encoder.num_layers - 1:
                    h = F.relu(h)
        return h


class DynamicGraphEncoder(nn.Module):
    """
    Encoder for temporal sequences of graphs.
    Captures evolution of service dependencies over time.
    """

    def __init__(self,
                 in_channels: int = 6,
                 hidden_channels: int = 64,
                 graph_out_channels: int = 128,
                 temporal_hidden: int = 64,
                 out_channels: int = 128,
                 architecture: str = 'gat',
                 dropout: float = 0.1):
        """
        Args:
            in_channels: Input node feature dimension
            hidden_channels: Hidden layer dimension
            graph_out_channels: Output dimension of graph encoder
            temporal_hidden: Hidden dimension for temporal modeling
            out_channels: Final output dimension
            architecture: GNN architecture ('gat' or 'gcn')
            dropout: Dropout rate
        """
        super().__init__()

        # Graph encoder for each snapshot
        self.graph_encoder = GraphEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=graph_out_channels,
            architecture=architecture,
            dropout=dropout
        )

        # Temporal encoder (GRU over graph snapshots)
        self.temporal_encoder = nn.GRU(
            input_size=graph_out_channels,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(temporal_hidden * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )

        self.output_dim = out_channels

    def forward(self, graph_sequence: List['Data']) -> torch.Tensor:
        """
        Forward pass for sequence of graphs.

        Args:
            graph_sequence: List of PyG Data objects

        Returns:
            (1, out_channels) dynamic graph embedding
        """
        # Encode each graph snapshot
        graph_embeddings = []
        for graph in graph_sequence:
            h = self.graph_encoder(
                graph.x, graph.edge_index,
                edge_attr=graph.edge_attr if hasattr(graph, 'edge_attr') else None
            )
            graph_embeddings.append(h)

        # Stack into sequence
        seq = torch.stack(graph_embeddings, dim=1)  # (1, num_graphs, graph_dim)

        # Temporal encoding
        output, h_n = self.temporal_encoder(seq)

        # Use final hidden state
        h_final = torch.cat([h_n[0], h_n[1]], dim=-1)

        # Project to output
        return self.output_proj(h_final)


class GraphPooling(nn.Module):
    """
    Hierarchical graph pooling for multi-scale representations.
    """

    def __init__(self,
                 in_channels: int,
                 pool_ratio: float = 0.5):
        super().__init__()

        # Note: Requires torch_geometric SAGPooling or TopKPooling
        # This is a simplified version using attention-based soft pooling
        self.attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1)
        )
        self.pool_ratio = pool_ratio

    def forward(self,
                x: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention scores and perform soft pooling.

        Returns:
            (pooled_features, attention_scores)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)
        scores = torch.softmax(scores, dim=0)

        # Weighted sum
        pooled = (x * scores.unsqueeze(-1)).sum(dim=0, keepdim=True)

        return pooled, scores


def create_graph_from_trace(spans_df,
                            service_vocab: Dict[str, int],
                            max_nodes: int = 50) -> 'Data':
    """
    Create PyG Data object from a single trace's spans.

    Args:
        spans_df: DataFrame with span data for one trace
        service_vocab: Mapping from service name to index
        max_nodes: Maximum number of nodes

    Returns:
        PyTorch Geometric Data object
    """
    if not HAS_TORCH_GEOMETRIC:
        raise ImportError("torch_geometric required")

    # Sort by start time
    spans_df = spans_df.sort_values('start_time')

    # Limit nodes
    if len(spans_df) > max_nodes:
        spans_df = spans_df.head(max_nodes)

    # Node features: [duration_ms, is_root, service_idx, relative_time, ...]
    node_features = []
    span_ids = spans_df['span_id'].tolist()
    span_to_idx = {sid: i for i, sid in enumerate(span_ids)}

    start_time_base = spans_df['start_time'].min()

    for _, row in spans_df.iterrows():
        is_root = 1.0 if row.get('parent_span_id') is None else 0.0
        service_idx = service_vocab.get(row['service_name'], 0)
        relative_time = (row['start_time'] - start_time_base) / 1e6  # Convert to seconds

        features = [
            row['duration_ms'] / 1000,  # Normalize to seconds
            is_root,
            float(service_idx),
            relative_time,
            1.0,  # Placeholder
            0.0   # Placeholder
        ]
        node_features.append(features)

    x = torch.tensor(node_features, dtype=torch.float)

    # Build edges from parent-child relationships
    edges = []
    for _, row in spans_df.iterrows():
        parent_id = row.get('parent_span_id')
        if parent_id and parent_id in span_to_idx:
            src_idx = span_to_idx[parent_id]
            dst_idx = span_to_idx[row['span_id']]
            edges.append([src_idx, dst_idx])

    # If no parent relationships, use temporal ordering
    if not edges:
        for i in range(len(span_ids) - 1):
            edges.append([i, i + 1])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.ones(len(edges), 1, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
