"""
Service Graph Builder Module
Constructs service dependency graphs from microservice traces.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch_geometric not installed. Some features will be unavailable.")


@dataclass
class ServiceGraph:
    """Represents a service dependency graph."""
    nodes: List[str]  # Service names
    edges: List[Tuple[str, str]]  # (source, target) pairs
    edge_weights: Dict[Tuple[str, str], float]
    node_features: Optional[np.ndarray] = None
    edge_features: Optional[np.ndarray] = None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for (src, dst), weight in self.edge_weights.items():
            G.add_edge(src, dst, weight=weight)
        return G

    def to_pyg_data(self) -> 'Data':
        """Convert to PyTorch Geometric Data object."""
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch_geometric required for this method")

        # Create node index mapping
        node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        # Create edge index tensor
        edge_index = torch.tensor([
            [node_to_idx[src], node_to_idx[dst]]
            for src, dst in self.edges
        ], dtype=torch.long).t().contiguous()

        # Create edge weights tensor
        edge_attr = torch.tensor([
            self.edge_weights.get((src, dst), 1.0)
            for src, dst in self.edges
        ], dtype=torch.float).unsqueeze(1)

        # Create node features (if available, else use degree)
        if self.node_features is not None:
            x = torch.tensor(self.node_features, dtype=torch.float)
        else:
            # Use in-degree and out-degree as default features
            G = self.to_networkx()
            x = torch.tensor([
                [G.in_degree(n), G.out_degree(n)]
                for n in self.nodes
            ], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class ServiceGraphBuilder:
    """Builds service dependency graphs from trace data."""

    def __init__(self):
        self.global_graph = None
        self.service_stats = defaultdict(lambda: {
            'call_count': 0,
            'avg_latency': 0.0,
            'error_count': 0
        })

    def build_from_traces(self, df_traces: pd.DataFrame,
                          add_self_loops: bool = False) -> ServiceGraph:
        """
        Build service dependency graph from trace DataFrame.

        Args:
            df_traces: DataFrame with columns [trace_id, service_name, start_time, ...]
            add_self_loops: Whether to add self-loop edges

        Returns:
            ServiceGraph object
        """
        # Get unique services
        services = df_traces['service_name'].unique().tolist()
        service_to_idx = {s: i for i, s in enumerate(services)}

        # Build edges from trace sequences
        edge_counts = defaultdict(int)

        for trace_id, group in df_traces.groupby('trace_id'):
            # Sort by start time to get call sequence
            sorted_group = group.sort_values('start_time')
            service_seq = sorted_group['service_name'].tolist()

            # Create edges between consecutive services
            for i in range(len(service_seq) - 1):
                src, dst = service_seq[i], service_seq[i + 1]
                if add_self_loops or src != dst:
                    edge_counts[(src, dst)] += 1

        # Normalize edge weights
        max_count = max(edge_counts.values()) if edge_counts else 1
        edge_weights = {k: v / max_count for k, v in edge_counts.items()}

        # Build edges list
        edges = list(edge_counts.keys())

        # Compute node features (latency statistics per service)
        node_features = self._compute_node_features(df_traces, services)

        return ServiceGraph(
            nodes=services,
            edges=edges,
            edge_weights=edge_weights,
            node_features=node_features
        )

    def _compute_node_features(self, df: pd.DataFrame,
                                services: List[str]) -> np.ndarray:
        """Compute node features for each service."""
        features = []

        for service in services:
            service_df = df[df['service_name'] == service]

            if len(service_df) == 0:
                features.append([0.0] * 6)
                continue

            # Latency features
            latencies = service_df['duration_ms'].values
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies) if len(latencies) > 1 else 0.0
            max_latency = np.max(latencies)

            # Call count (normalized)
            call_count = len(service_df)

            # Unique traces this service appears in
            trace_count = service_df['trace_id'].nunique()

            # Average spans per trace for this service
            spans_per_trace = call_count / max(trace_count, 1)

            features.append([
                avg_latency / 1000,  # Normalize to seconds
                std_latency / 1000,
                max_latency / 1000,
                np.log1p(call_count),  # Log-normalized call count
                np.log1p(trace_count),
                spans_per_trace
            ])

        return np.array(features, dtype=np.float32)

    def build_dynamic_graph(self, df_traces: pd.DataFrame,
                            window_size: int = 100) -> List[ServiceGraph]:
        """
        Build sequence of dynamic graphs over time windows.

        Args:
            df_traces: Trace DataFrame sorted by time
            window_size: Number of traces per window

        Returns:
            List of ServiceGraph objects for each time window
        """
        graphs = []
        trace_ids = df_traces['trace_id'].unique()

        for i in range(0, len(trace_ids), window_size):
            window_traces = trace_ids[i:i + window_size]
            window_df = df_traces[df_traces['trace_id'].isin(window_traces)]
            graph = self.build_from_traces(window_df)
            graphs.append(graph)

        return graphs

    def build_trace_graph(self, df_single_trace: pd.DataFrame) -> ServiceGraph:
        """
        Build graph for a single trace (span-level graph).

        Args:
            df_single_trace: DataFrame containing spans of a single trace

        Returns:
            ServiceGraph for this trace
        """
        # Use span_id as nodes for fine-grained graph
        spans = df_single_trace.sort_values('start_time')

        nodes = spans['span_id'].tolist()
        services = spans['service_name'].tolist()

        # Build edges from parent-child relationships
        edges = []
        edge_weights = {}

        span_to_service = dict(zip(spans['span_id'], spans['service_name']))

        if 'parent_span_id' in spans.columns:
            for _, row in spans.iterrows():
                if pd.notna(row['parent_span_id']) and row['parent_span_id'] in span_to_service:
                    edge = (row['parent_span_id'], row['span_id'])
                    edges.append(edge)
                    edge_weights[edge] = 1.0

        # If no parent info, use temporal ordering
        if not edges:
            for i in range(len(nodes) - 1):
                edge = (nodes[i], nodes[i + 1])
                edges.append(edge)
                edge_weights[edge] = 1.0

        # Node features: [latency, is_root, depth_estimate]
        node_features = []
        for _, row in spans.iterrows():
            is_root = 1.0 if pd.isna(row.get('parent_span_id')) else 0.0
            node_features.append([
                row['duration_ms'] / 1000,
                is_root,
                0.0  # Placeholder for depth
            ])

        return ServiceGraph(
            nodes=nodes,
            edges=edges,
            edge_weights=edge_weights,
            node_features=np.array(node_features, dtype=np.float32)
        )


def visualize_service_graph(graph: ServiceGraph, output_path: Optional[str] = None):
    """Visualize service dependency graph."""
    import matplotlib.pyplot as plt

    G = graph.to_networkx()

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node sizes based on in-degree
    node_sizes = [300 + G.in_degree(n) * 100 for n in G.nodes()]

    # Edge widths based on weight
    edge_widths = [graph.edge_weights.get((u, v), 0.5) * 3 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           alpha=0.6, edge_color='gray', arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title('Service Dependency Graph', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format='svg', bbox_inches='tight')
    plt.show()
