"""
Trace Preprocessing Module
Data preprocessing and feature extraction for microservice traces.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from collections import defaultdict


class TracePreprocessor:
    """
    Comprehensive preprocessor for microservice trace data.
    Handles normalization, encoding, and feature extraction.
    """

    def __init__(self,
                 max_seq_len: int = 50,
                 latency_scaler: str = 'standard',
                 handle_missing: str = 'zero'):
        """
        Args:
            max_seq_len: Maximum sequence length for padding/truncation
            latency_scaler: 'standard', 'minmax', or 'log'
            handle_missing: 'zero', 'mean', or 'drop'
        """
        self.max_seq_len = max_seq_len
        self.latency_scaler_type = latency_scaler
        self.handle_missing = handle_missing

        # Encoders and scalers
        self.service_encoder = LabelEncoder()
        self.operation_encoder = LabelEncoder()

        if latency_scaler == 'standard':
            self.latency_scaler = StandardScaler()
        elif latency_scaler == 'minmax':
            self.latency_scaler = MinMaxScaler()
        else:
            self.latency_scaler = None

        # Fitted state
        self._is_fitted = False
        self.service_vocab: Dict[str, int] = {}
        self.operation_vocab: Dict[str, int] = {}

        # Statistics
        self.stats = {}

    def fit(self, df: pd.DataFrame) -> 'TracePreprocessor':
        """
        Fit preprocessor on training data.

        Args:
            df: DataFrame with columns [trace_id, span_id, service_name,
                operation_name, start_time, duration_ms, ...]
        """
        # Handle missing values
        df = self._handle_missing_values(df)

        # Fit service encoder
        services = df['service_name'].unique().tolist()
        services = ['<PAD>', '<UNK>'] + sorted([s for s in services if s not in ['<PAD>', '<UNK>']])
        self.service_encoder.fit(services)
        self.service_vocab = {s: i for i, s in enumerate(services)}

        # Fit operation encoder if available
        if 'operation_name' in df.columns:
            operations = df['operation_name'].unique().tolist()
            operations = ['<PAD>', '<UNK>'] + sorted([o for o in operations if o not in ['<PAD>', '<UNK>']])
            self.operation_encoder.fit(operations)
            self.operation_vocab = {o: i for i, o in enumerate(operations)}

        # Fit latency scaler
        if self.latency_scaler is not None:
            latencies = df['duration_ms'].values.reshape(-1, 1)
            if self.latency_scaler_type == 'log':
                latencies = np.log1p(latencies)
            self.latency_scaler.fit(latencies)

        # Compute statistics
        self.stats = {
            'n_traces': df['trace_id'].nunique(),
            'n_services': len(services) - 2,  # Exclude PAD and UNK
            'n_spans': len(df),
            'avg_latency': df['duration_ms'].mean(),
            'std_latency': df['duration_ms'].std(),
            'max_latency': df['duration_ms'].max(),
            'min_latency': df['duration_ms'].min()
        }

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Transform trace data into model-ready format.

        Args:
            df: DataFrame with trace data

        Returns:
            Dictionary with processed arrays and metadata
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        df = self._handle_missing_values(df)

        # Process each trace
        traces_data = []

        for trace_id, group in df.groupby('trace_id'):
            trace_data = self._process_single_trace(trace_id, group)
            traces_data.append(trace_data)

        # Aggregate into arrays
        result = self._aggregate_traces(traces_data)
        return result

    def fit_transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        df = df.copy()

        if self.handle_missing == 'drop':
            df = df.dropna(subset=['service_name', 'duration_ms'])
        elif self.handle_missing == 'zero':
            df['duration_ms'] = df['duration_ms'].fillna(0)
            df['service_name'] = df['service_name'].fillna('<UNK>')
        elif self.handle_missing == 'mean':
            df['duration_ms'] = df['duration_ms'].fillna(df['duration_ms'].mean())
            df['service_name'] = df['service_name'].fillna('<UNK>')

        return df

    def _process_single_trace(self, trace_id: str, group: pd.DataFrame) -> Dict:
        """Process a single trace into features."""
        # Sort by start time
        group = group.sort_values('start_time')

        # Extract service sequence
        services = group['service_name'].tolist()
        service_ids = [
            self.service_vocab.get(s, self.service_vocab.get('<UNK>', 1))
            for s in services
        ]

        # Extract latencies
        latencies = group['duration_ms'].tolist()

        # Pad or truncate
        if len(service_ids) > self.max_seq_len:
            service_ids = service_ids[:self.max_seq_len]
            latencies = latencies[:self.max_seq_len]
        else:
            pad_len = self.max_seq_len - len(service_ids)
            service_ids = service_ids + [0] * pad_len
            latencies = latencies + [0.0] * pad_len

        # Scale latencies
        latencies_arr = np.array(latencies, dtype=np.float32)
        if self.latency_scaler is not None:
            # Only scale non-zero values
            non_zero_mask = latencies_arr > 0
            if non_zero_mask.any():
                latencies_arr[non_zero_mask] = self.latency_scaler.transform(
                    latencies_arr[non_zero_mask].reshape(-1, 1)
                ).flatten()

        # Extract additional features
        trace_features = self._extract_trace_features(group)

        return {
            'trace_id': trace_id,
            'service_ids': np.array(service_ids, dtype=np.int64),
            'latencies': latencies_arr,
            'services': services[:self.max_seq_len],
            'trace_features': trace_features,
            'num_spans': len(group),
            'total_duration': group['duration_ms'].sum()
        }

    def _extract_trace_features(self, group: pd.DataFrame) -> np.ndarray:
        """Extract aggregate trace-level features."""
        features = []

        # Latency statistics
        latencies = group['duration_ms'].values
        features.extend([
            np.mean(latencies),
            np.std(latencies) if len(latencies) > 1 else 0,
            np.max(latencies),
            np.min(latencies),
            np.median(latencies)
        ])

        # Span count
        features.append(len(group))

        # Unique services
        features.append(group['service_name'].nunique())

        # Time span
        if len(group) > 1:
            time_span = group['start_time'].max() - group['start_time'].min()
            features.append(time_span / 1e6)  # Convert to seconds
        else:
            features.append(0)

        return np.array(features, dtype=np.float32)

    def _aggregate_traces(self, traces_data: List[Dict]) -> Dict[str, Any]:
        """Aggregate processed traces into arrays."""
        n_traces = len(traces_data)

        # Stack arrays
        service_ids = np.stack([t['service_ids'] for t in traces_data])
        latencies = np.stack([t['latencies'] for t in traces_data])
        trace_features = np.stack([t['trace_features'] for t in traces_data])

        # Collect metadata
        trace_ids = [t['trace_id'] for t in traces_data]
        service_sequences = [t['services'] for t in traces_data]

        return {
            'service_ids': service_ids,
            'latencies': latencies,
            'trace_features': trace_features,
            'trace_ids': trace_ids,
            'service_sequences': service_sequences,
            'service_vocab': self.service_vocab,
            'vocab_size': len(self.service_vocab),
            'n_traces': n_traces
        }

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.service_vocab)

    def get_service_vocab(self) -> Dict[str, int]:
        """Get service vocabulary."""
        return self.service_vocab.copy()

    def decode_sequence(self, sequence: np.ndarray) -> List[str]:
        """Decode service ID sequence back to names."""
        idx_to_service = {v: k for k, v in self.service_vocab.items()}
        return [idx_to_service.get(idx, '<UNK>') for idx in sequence if idx > 0]


class FeatureEngineer:
    """
    Advanced feature engineering for trace data.
    """

    def __init__(self):
        self.feature_names = []

    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal patterns from trace data."""
        features = []

        for trace_id, group in df.groupby('trace_id'):
            group = group.sort_values('start_time')

            # Inter-arrival times
            start_times = group['start_time'].values
            if len(start_times) > 1:
                inter_arrivals = np.diff(start_times)
                iat_mean = np.mean(inter_arrivals)
                iat_std = np.std(inter_arrivals)
                iat_max = np.max(inter_arrivals)
            else:
                iat_mean = iat_std = iat_max = 0

            # Service switching patterns
            services = group['service_name'].tolist()
            switches = sum(1 for i in range(1, len(services)) if services[i] != services[i-1])
            switch_rate = switches / max(len(services) - 1, 1)

            # Latency patterns
            latencies = group['duration_ms'].values
            latency_trend = np.polyfit(range(len(latencies)), latencies, 1)[0] if len(latencies) > 1 else 0

            features.append({
                'trace_id': trace_id,
                'iat_mean': iat_mean,
                'iat_std': iat_std,
                'iat_max': iat_max,
                'service_switch_rate': switch_rate,
                'latency_trend': latency_trend,
                'num_unique_services': group['service_name'].nunique(),
                'total_spans': len(group)
            })

        return pd.DataFrame(features)

    def extract_graph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract graph-based features from service dependencies."""
        import networkx as nx

        features = []

        for trace_id, group in df.groupby('trace_id'):
            group = group.sort_values('start_time')
            services = group['service_name'].tolist()

            # Build trace graph
            G = nx.DiGraph()
            for i in range(len(services) - 1):
                src, dst = services[i], services[i + 1]
                if G.has_edge(src, dst):
                    G[src][dst]['weight'] += 1
                else:
                    G.add_edge(src, dst, weight=1)

            # Graph metrics
            if len(G) > 0:
                density = nx.density(G)
                try:
                    avg_clustering = nx.average_clustering(G.to_undirected())
                except:
                    avg_clustering = 0
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
            else:
                density = avg_clustering = 0
                n_nodes = n_edges = 0

            features.append({
                'trace_id': trace_id,
                'graph_density': density,
                'avg_clustering': avg_clustering,
                'num_nodes': n_nodes,
                'num_edges': n_edges
            })

        return pd.DataFrame(features)


class AnomalyLabeler:
    """
    Automatic anomaly labeling based on heuristics.
    Useful when ground truth labels are not available.
    """

    def __init__(self,
                 latency_threshold: float = 3.0,
                 error_keywords: List[str] = None):
        """
        Args:
            latency_threshold: Z-score threshold for latency anomalies
            error_keywords: Keywords indicating errors in logs/tags
        """
        self.latency_threshold = latency_threshold
        self.error_keywords = error_keywords or ['error', 'fail', 'timeout', 'exception']

    def label_traces(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label traces as normal (0) or anomaly (1).

        Args:
            df: DataFrame with trace data

        Returns:
            DataFrame with 'is_anomaly' column added
        """
        df = df.copy()

        # Compute trace-level labels
        trace_labels = {}

        for trace_id, group in df.groupby('trace_id'):
            is_anomaly = self._check_anomaly(group)
            trace_labels[trace_id] = is_anomaly

        # Map back to spans
        df['is_anomaly'] = df['trace_id'].map(trace_labels).astype(int)

        return df

    def _check_anomaly(self, group: pd.DataFrame) -> bool:
        """Check if a trace is anomalous."""
        # Latency-based check
        latencies = group['duration_ms'].values
        if len(latencies) > 0:
            mean_lat = np.mean(latencies)
            std_lat = np.std(latencies) if len(latencies) > 1 else 1

            # Check for extreme latencies
            z_scores = np.abs((latencies - mean_lat) / (std_lat + 1e-8))
            if np.any(z_scores > self.latency_threshold):
                return True

        # Error keyword check (if tags available)
        if 'tags' in group.columns:
            for tags in group['tags'].dropna():
                if isinstance(tags, str):
                    tags_lower = tags.lower()
                    if any(kw in tags_lower for kw in self.error_keywords):
                        return True

        # Status code check (if available)
        if 'status_code' in group.columns:
            status_codes = group['status_code'].dropna().values
            if any(code >= 400 for code in status_codes):
                return True

        return False


class DataAugmenter:
    """
    Data augmentation for trace sequences.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def augment_latencies(self,
                          latencies: np.ndarray,
                          noise_factor: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to latencies."""
        noise = self.rng.normal(0, noise_factor, latencies.shape)
        augmented = latencies * (1 + noise)
        return np.clip(augmented, 0, None)

    def augment_sequence(self,
                         sequence: np.ndarray,
                         swap_prob: float = 0.1) -> np.ndarray:
        """Randomly swap adjacent elements in sequence."""
        augmented = sequence.copy()
        n = len(augmented)

        for i in range(n - 1):
            if self.rng.random() < swap_prob and augmented[i] != 0 and augmented[i + 1] != 0:
                augmented[i], augmented[i + 1] = augmented[i + 1], augmented[i]

        return augmented

    def generate_synthetic_anomaly(self,
                                    normal_trace: Dict,
                                    anomaly_type: str = 'latency') -> Dict:
        """Generate synthetic anomaly from normal trace."""
        anomaly = {k: v.copy() if isinstance(v, np.ndarray) else v
                   for k, v in normal_trace.items()}

        if anomaly_type == 'latency':
            # Inject latency spike
            idx = self.rng.randint(0, len(anomaly['latencies']))
            anomaly['latencies'][idx] *= self.rng.uniform(5, 20)

        elif anomaly_type == 'sequence':
            # Inject service order violation
            anomaly['service_ids'] = self.augment_sequence(
                anomaly['service_ids'], swap_prob=0.3
            )

        elif anomaly_type == 'missing':
            # Simulate missing span
            idx = self.rng.randint(1, len(anomaly['service_ids']))
            anomaly['service_ids'][idx] = 0
            anomaly['latencies'][idx] = 0

        return anomaly
