"""
Dataset Module
PyTorch Dataset classes for microservice trace anomaly detection.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

from .trace_parser import Trace, extract_call_sequences
from .graph_builder import ServiceGraph, ServiceGraphBuilder


class MicroserviceTraceDataset(Dataset):
    """
    PyTorch Dataset for microservice trace anomaly detection.

    Each sample contains:
    - Syntactic features (SAX-encoded patterns)
    - Graph features (service dependency subgraph)
    - Temporal features (latency sequences)
    - Label (0: normal, 1: anomaly)
    """

    def __init__(self,
                 traces: List[Trace],
                 labels: Optional[np.ndarray] = None,
                 service_vocab: Optional[Dict[str, int]] = None,
                 max_seq_len: int = 50,
                 transform=None):
        """
        Args:
            traces: List of Trace objects
            labels: Array of labels (0/1) for each trace
            service_vocab: Service name to index mapping
            max_seq_len: Maximum sequence length for padding
            transform: Optional transform function
        """
        self.traces = traces
        self.labels = labels if labels is not None else np.zeros(len(traces))
        self.max_seq_len = max_seq_len
        self.transform = transform

        # Build service vocabulary if not provided
        if service_vocab is None:
            all_services = set()
            for trace in traces:
                all_services.update(trace.services)
            self.service_vocab = {s: i + 1 for i, s in enumerate(sorted(all_services))}
            self.service_vocab['<PAD>'] = 0
            self.service_vocab['<UNK>'] = len(self.service_vocab)
        else:
            self.service_vocab = service_vocab

        self.vocab_size = len(self.service_vocab)

        # Precompute features
        self._precompute_features()

    def _precompute_features(self):
        """Precompute features for all traces."""
        self.sequences = []
        self.latency_seqs = []

        for trace in self.traces:
            # Service call sequence
            call_seq = trace.get_call_sequence()
            encoded_seq = [self.service_vocab.get(s, self.service_vocab['<UNK>'])
                          for s in call_seq]

            # Pad or truncate
            if len(encoded_seq) > self.max_seq_len:
                encoded_seq = encoded_seq[:self.max_seq_len]
            else:
                encoded_seq = encoded_seq + [0] * (self.max_seq_len - len(encoded_seq))

            self.sequences.append(encoded_seq)

            # Latency sequence
            latencies = [s.duration_ms for s in sorted(trace.spans, key=lambda x: x.start_time)]
            if len(latencies) > self.max_seq_len:
                latencies = latencies[:self.max_seq_len]
            else:
                latencies = latencies + [0.0] * (self.max_seq_len - len(latencies))

            self.latency_seqs.append(latencies)

        self.sequences = np.array(self.sequences, dtype=np.int64)
        self.latency_seqs = np.array(self.latency_seqs, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.traces)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = {
            'sequence': torch.tensor(self.sequences[idx], dtype=torch.long),
            'latencies': torch.tensor(self.latency_seqs[idx], dtype=torch.float32),
            'label': torch.tensor(self.labels[idx], dtype=torch.float32),
            'trace_id': self.traces[idx].trace_id,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_service_vocab(self) -> Dict[str, int]:
        return self.service_vocab


class TraceSequenceDataset(Dataset):
    """
    Simplified dataset for sequence-based models.
    """

    def __init__(self,
                 sequences: np.ndarray,
                 labels: np.ndarray,
                 latencies: Optional[np.ndarray] = None):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        if latencies is not None:
            self.latencies = torch.tensor(latencies, dtype=torch.float32)
        else:
            self.latencies = None

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }
        if self.latencies is not None:
            item['latencies'] = self.latencies[idx]
        return item


def create_dataloaders(dataset: MicroserviceTraceDataset,
                       batch_size: int = 32,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       random_state: int = 42) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with stratified split.

    Args:
        dataset: Full dataset
        batch_size: Batch size for all loaders
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        random_state: Random seed

    Returns:
        (train_loader, val_loader, test_loader)
    """
    n = len(dataset)
    indices = np.arange(n)
    labels = dataset.labels

    # First split: train vs (val + test)
    train_idx, temp_idx = train_test_split(
        indices, train_size=train_ratio,
        stratify=labels, random_state=random_state
    )

    # Second split: val vs test
    val_size = val_ratio / (1 - train_ratio)
    temp_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=val_size,
        stratify=temp_labels, random_state=random_state
    )

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=0
    )

    print(f"Dataset splits: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    return train_loader, val_loader, test_loader


def prepare_data_from_df(df: pd.DataFrame,
                         label_col: Optional[str] = 'is_anomaly',
                         max_seq_len: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare data arrays from trace DataFrame.

    Args:
        df: DataFrame with trace data
        label_col: Column name for labels
        max_seq_len: Maximum sequence length

    Returns:
        (sequences, latencies, labels, service_vocab)
    """
    # Build vocabulary
    services = df['service_name'].unique().tolist()
    service_vocab = {s: i + 1 for i, s in enumerate(sorted(services))}
    service_vocab['<PAD>'] = 0
    service_vocab['<UNK>'] = len(service_vocab)

    sequences = []
    latencies = []
    labels = []

    for trace_id, group in df.groupby('trace_id'):
        sorted_group = group.sort_values('start_time')

        # Encode service sequence
        seq = [service_vocab.get(s, service_vocab['<UNK>'])
               for s in sorted_group['service_name']]
        lat = sorted_group['duration_ms'].tolist()

        # Pad/truncate
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
            lat = lat[:max_seq_len]
        else:
            seq = seq + [0] * (max_seq_len - len(seq))
            lat = lat + [0.0] * (max_seq_len - len(lat))

        sequences.append(seq)
        latencies.append(lat)

        # Get label (majority vote if multiple spans)
        if label_col and label_col in group.columns:
            label = group[label_col].mode().iloc[0] if len(group) > 0 else 0
        else:
            label = 0
        labels.append(label)

    return (np.array(sequences, dtype=np.int64),
            np.array(latencies, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            service_vocab)
