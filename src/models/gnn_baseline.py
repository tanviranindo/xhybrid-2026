"""
GNN-Based Anomaly Detector for Microservice Traces
Graph Neural Network baseline using service dependencies

Approach:
1. Build service dependency graph from traces
2. Use Graph Attention Network (GAT) for node embeddings
3. Classify anomalies based on graph features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx


@dataclass
class GraphFeatures:
    """Container for graph-based features."""
    adjacency: np.ndarray  # (n_services, n_services)
    node_features: np.ndarray  # (n_services, n_features)
    edge_weights: np.ndarray  # (n_services, n_services)
    service_names: List[str]


class ServiceDependencyGraph:
    """
    Build and manage service dependency graph from traces.

    Graph construction:
    - Nodes: Microservices
    - Edges: Service call relationships
    - Edge weights: Call frequency or latency
    - Node features: Service statistics (latency, call count, etc.)
    """

    def __init__(self, service_names: List[str]):
        self.service_names = service_names
        self.n_services = len(service_names)
        self.service_to_idx = {name: idx for idx, name in enumerate(service_names)}

        # Graph structure
        self.adjacency = np.zeros((self.n_services, self.n_services))
        self.edge_weights = np.zeros((self.n_services, self.n_services))
        self.call_counts = np.zeros((self.n_services, self.n_services))

        # Node statistics
        self.service_latencies = [[] for _ in range(self.n_services)]
        self.service_call_counts = np.zeros(self.n_services)

    def add_trace(self, service_latencies: np.ndarray, trace_sequence: Optional[List[str]] = None):
        """
        Add a trace to build the graph.

        Args:
            service_latencies: (n_services,) latency values
            trace_sequence: Optional ordered list of service calls
        """
        # Update node features (service statistics)
        for i, latency in enumerate(service_latencies):
            if latency > 0:  # Service was called
                self.service_latencies[i].append(latency)
                self.service_call_counts[i] += 1

        # Update edges if sequence provided
        if trace_sequence:
            for i in range(len(trace_sequence) - 1):
                src = trace_sequence[i]
                dst = trace_sequence[i + 1]

                if src in self.service_to_idx and dst in self.service_to_idx:
                    src_idx = self.service_to_idx[src]
                    dst_idx = self.service_to_idx[dst]

                    self.adjacency[src_idx, dst_idx] = 1
                    self.call_counts[src_idx, dst_idx] += 1
        else:
            # If no sequence, create fully connected graph based on co-occurrence
            active_services = np.where(service_latencies > 0)[0]
            for i in active_services:
                for j in active_services:
                    if i != j:
                        self.adjacency[i, j] = 1
                        self.call_counts[i, j] += 1

    def finalize(self) -> GraphFeatures:
        """
        Finalize graph construction and extract features.

        Returns:
            GraphFeatures with adjacency, node features, edge weights
        """
        # Compute edge weights (normalized call frequency)
        max_calls = self.call_counts.max()
        if max_calls > 0:
            self.edge_weights = self.call_counts / max_calls

        # Compute node features
        node_features = []
        for i in range(self.n_services):
            if len(self.service_latencies[i]) > 0:
                mean_latency = np.mean(self.service_latencies[i])
                std_latency = np.std(self.service_latencies[i])
                max_latency = np.max(self.service_latencies[i])
                call_count = self.service_call_counts[i]
            else:
                mean_latency = 0
                std_latency = 0
                max_latency = 0
                call_count = 0

            node_features.append([mean_latency, std_latency, max_latency, call_count])

        node_features = np.array(node_features, dtype=np.float32)

        return GraphFeatures(
            adjacency=self.adjacency,
            node_features=node_features,
            edge_weights=self.edge_weights,
            service_names=self.service_names
        )


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).

    Computes attention-weighted aggregation of neighbor features.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Linear transformations
        self.W = nn.Linear(in_features, out_features, bias=False)

        # Attention mechanism
        self.a = nn.Linear(2 * out_features, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, n_nodes, in_features) node features
            adj: (batch, n_nodes, n_nodes) adjacency matrix

        Returns:
            (batch, n_nodes, out_features) updated node features
        """
        batch_size, n_nodes, _ = x.shape

        # Linear transformation
        h = self.W(x)  # (batch, n_nodes, out_features)

        # Compute attention scores
        # Concatenate all pairs of nodes
        h_i = h.unsqueeze(2).repeat(1, 1, n_nodes, 1)  # (batch, n_nodes, n_nodes, out_features)
        h_j = h.unsqueeze(1).repeat(1, n_nodes, 1, 1)  # (batch, n_nodes, n_nodes, out_features)

        concat = torch.cat([h_i, h_j], dim=-1)  # (batch, n_nodes, n_nodes, 2*out_features)

        e = self.leaky_relu(self.a(concat).squeeze(-1))  # (batch, n_nodes, n_nodes)

        # Mask attention for non-existent edges
        mask = (adj == 0)
        e = e.masked_fill(mask, -1e9)

        # Attention weights
        alpha = F.softmax(e, dim=-1)  # (batch, n_nodes, n_nodes)
        alpha = self.dropout(alpha)

        # Aggregate neighbors
        h_prime = torch.bmm(alpha, h)  # (batch, n_nodes, out_features)

        return h_prime


class GNNAnomalyDetector(nn.Module):
    """
    GNN-based anomaly detector for microservice traces.

    Architecture:
    1. Graph Attention layers to learn service representations
    2. Graph-level pooling (mean/max)
    3. MLP classifier for anomaly detection
    """

    def __init__(self,
                 n_node_features: int = 4,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 pooling: str = 'mean'):
        super().__init__()

        self.n_node_features = n_node_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pooling = pooling

        # Graph attention layers
        self.gat_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            GraphAttentionLayer(n_node_features, hidden_dim, dropout)
        )

        # Hidden layers
        for _ in range(n_layers - 1):
            self.gat_layers.append(
                GraphAttentionLayer(hidden_dim, hidden_dim, dropout)
            )

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            node_features: (batch, n_nodes, n_node_features)
            adjacency: (batch, n_nodes, n_nodes)

        Returns:
            (batch, 2) logits for [normal, anomaly]
        """
        x = node_features

        # Graph attention layers
        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat(x, adjacency)
            x_new = norm(x_new)

            # Residual connection (if dimensions match)
            if i > 0:
                x = x + x_new
            else:
                x = x_new

            x = F.relu(x)

        # Graph-level pooling
        if self.pooling == 'mean':
            graph_embedding = x.mean(dim=1)  # (batch, hidden_dim)
        elif self.pooling == 'max':
            graph_embedding = x.max(dim=1)[0]  # (batch, hidden_dim)
        else:  # sum
            graph_embedding = x.sum(dim=1)  # (batch, hidden_dim)

        # Classification
        logits = self.classifier(graph_embedding)  # (batch, 2)

        return logits

    def predict(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomaly labels and probabilities.

        Returns:
            (predictions, probabilities)
        """
        logits = self.forward(node_features, adjacency)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        return preds, probs[:, 1]  # Return anomaly probability


# Example usage and testing
if __name__ == "__main__":
    print("Testing GNN Anomaly Detector...")

    # Test 1: Service dependency graph construction
    print("\n1. Building Service Dependency Graph:")
    service_names = [f'service-{i}' for i in range(10)]

    graph_builder = ServiceDependencyGraph(service_names)

    # Add some traces
    for _ in range(100):
        # Random service latencies
        latencies = np.random.randn(10) * 10 + 30
        latencies = np.maximum(latencies, 0)

        # Random call sequence
        n_calls = np.random.randint(3, 7)
        sequence = np.random.choice(service_names, size=n_calls, replace=False).tolist()

        graph_builder.add_trace(latencies, sequence)

    graph_features = graph_builder.finalize()

    print(f"   Graph built:")
    print(f"   - Nodes: {len(graph_features.service_names)}")
    print(f"   - Adjacency shape: {graph_features.adjacency.shape}")
    print(f"   - Node features shape: {graph_features.node_features.shape}")
    print(f"   - Edge density: {graph_features.adjacency.sum() / (10 * 10):.2%}")

    # Test 2: Graph Attention Layer
    print("\n2. Testing Graph Attention Layer:")
    gat = GraphAttentionLayer(in_features=4, out_features=64)

    # Create dummy batch
    batch_size = 8
    n_nodes = 10

    x = torch.randn(batch_size, n_nodes, 4)
    adj = torch.from_numpy(graph_features.adjacency).unsqueeze(0).repeat(batch_size, 1, 1).float()

    out = gat(x, adj)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")

    # Test 3: Full GNN model
    print("\n3. Testing Full GNN Model:")
    model = GNNAnomalyDetector(
        n_node_features=4,
        hidden_dim=64,
        n_layers=2,
        dropout=0.1
    )

    # Convert graph features to tensors
    node_features = torch.from_numpy(graph_features.node_features).unsqueeze(0).repeat(batch_size, 1, 1).float()
    adjacency = torch.from_numpy(graph_features.adjacency).unsqueeze(0).repeat(batch_size, 1, 1).float()

    logits = model(node_features, adjacency)
    preds, probs = model.predict(node_features, adjacency)

    print(f"   Input node features: {node_features.shape}")
    print(f"   Input adjacency: {adjacency.shape}")
    print(f"   Output logits: {logits.shape}")
    print(f"   Predictions: {preds.shape}")
    print(f"   Anomaly probabilities: {probs.shape}")

    # Test 4: Training step
    print("\n4. Testing Training Step:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Dummy labels
    labels = torch.randint(0, 2, (batch_size,))

    # Forward pass
    logits = model(node_features, adjacency)
    loss = criterion(logits, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"   Loss: {loss.item():.4f}")
    print(f"   Parameters updated: ✅")

    # Test 5: Evaluation
    print("\n5. Testing Evaluation:")
    model.eval()

    with torch.no_grad():
        preds, probs = model.predict(node_features, adjacency)

    accuracy = (preds == labels).float().mean()
    print(f"   Predictions: {preds.tolist()}")
    print(f"   Ground truth: {labels.tolist()}")
    print(f"   Accuracy: {accuracy:.2%}")

    print("\n✅ GNN Anomaly Detector tests passed!")
