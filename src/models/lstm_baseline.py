"""
LSTM-Based Anomaly Detector for Microservice Traces
Temporal sequence modeling baseline

Approach:
1. Treat service latencies as time series
2. Use BiLSTM to capture temporal patterns
3. Classify anomalies based on temporal features
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TemporalFeatures:
    """Container for temporal sequence features."""
    sequences: np.ndarray  # (n_samples, seq_len, n_features)
    lengths: np.ndarray  # (n_samples,) actual sequence lengths
    labels: Optional[np.ndarray] = None  # (n_samples,)


class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based anomaly detector for microservice traces.

    Architecture:
    1. BiLSTM layers to capture temporal patterns
    2. Attention mechanism over time steps
    3. MLP classifier for anomaly detection

    Args:
        input_dim: Number of input features (services)
        hidden_dim: LSTM hidden dimension
        n_layers: Number of LSTM layers
        dropout: Dropout rate
        use_attention: Whether to use attention mechanism
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 n_layers: int = 2,
                 dropout: float = 0.1,
                 use_attention: bool = True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.use_attention = use_attention

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )

        # Attention mechanism (if enabled)
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, input_dim) input sequences
            lengths: (batch,) actual sequence lengths (optional)

        Returns:
            (batch, 2) logits for [normal, anomaly]
        """
        batch_size, seq_len, _ = x.shape

        # LSTM encoding
        if lengths is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True, total_length=seq_len
            )
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        # lstm_out: (batch, seq_len, hidden_dim * 2)

        # Apply attention or use final hidden state
        if self.use_attention:
            # Compute attention weights
            attn_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = F.softmax(attn_scores, dim=1)  # (batch, seq_len, 1)

            # Weighted sum
            context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim * 2)
        else:
            # Use last hidden state (concatenate forward and backward)
            # h_n: (n_layers * 2, batch, hidden_dim)
            forward_h = h_n[-2]  # (batch, hidden_dim)
            backward_h = h_n[-1]  # (batch, hidden_dim)
            context = torch.cat([forward_h, backward_h], dim=-1)  # (batch, hidden_dim * 2)

        # Classification
        logits = self.classifier(context)  # (batch, 2)

        return logits

    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict anomaly labels and probabilities.

        Returns:
            (predictions, probabilities)
        """
        logits = self.forward(x, lengths)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        return preds, probs[:, 1]  # Return anomaly probability


class TemporalFeatureExtractor:
    """
    Extract temporal features from microservice traces.

    Converts static trace data into sequences suitable for LSTM.
    """

    def __init__(self, window_size: int = 10, stride: int = 1):
        """
        Args:
            window_size: Number of consecutive samples in each sequence
            stride: Step size for sliding window
        """
        self.window_size = window_size
        self.stride = stride

    def extract_sequences(self,
                         traces: np.ndarray,
                         labels: Optional[np.ndarray] = None) -> TemporalFeatures:
        """
        Extract sliding window sequences from traces.

        Args:
            traces: (n_samples, n_features) trace data
            labels: (n_samples,) optional labels

        Returns:
            TemporalFeatures with sequences
        """
        n_samples, n_features = traces.shape

        sequences = []
        seq_labels = []
        lengths = []

        # Sliding window
        for i in range(0, n_samples - self.window_size + 1, self.stride):
            window = traces[i:i + self.window_size]
            sequences.append(window)
            lengths.append(self.window_size)

            if labels is not None:
                # Label is the label of the last sample in the window
                seq_labels.append(labels[i + self.window_size - 1])

        sequences = np.array(sequences, dtype=np.float32)
        lengths = np.array(lengths, dtype=np.int64)

        if labels is not None:
            seq_labels = np.array(seq_labels, dtype=np.int64)
        else:
            seq_labels = None

        return TemporalFeatures(
            sequences=sequences,
            lengths=lengths,
            labels=seq_labels
        )

    def extract_single_sequence(self, trace: np.ndarray) -> np.ndarray:
        """
        Extract features from a single trace (no windowing).

        Args:
            trace: (n_features,) single trace

        Returns:
            (1, n_features) reshaped for LSTM
        """
        return trace.reshape(1, -1)


# Example usage and testing
if __name__ == "__main__":
    print("Testing LSTM Anomaly Detector...")

    # Test 1: LSTM model
    print("\n1. Testing LSTM Model:")
    model = LSTMAnomalyDetector(
        input_dim=14,  # 14 services
        hidden_dim=64,
        n_layers=2,
        dropout=0.1,
        use_attention=True
    )

    # Create dummy data
    batch_size = 8
    seq_len = 20
    input_dim = 14

    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.randint(10, seq_len + 1, (batch_size,))

    logits = model(x, lengths)
    preds, probs = model.predict(x, lengths)

    print(f"   Input shape: {x.shape}")
    print(f"   Lengths: {lengths.tolist()}")
    print(f"   Output logits: {logits.shape}")
    print(f"   Predictions: {preds.shape}")
    print(f"   Anomaly probabilities: {probs.shape}")

    # Test 2: Without attention
    print("\n2. Testing LSTM Without Attention:")
    model_no_attn = LSTMAnomalyDetector(
        input_dim=14,
        hidden_dim=64,
        n_layers=2,
        use_attention=False
    )

    logits = model_no_attn(x)
    print(f"   Output logits: {logits.shape}")

    # Test 3: Temporal feature extraction
    print("\n3. Testing Temporal Feature Extraction:")
    extractor = TemporalFeatureExtractor(window_size=10, stride=5)

    # Create dummy trace data
    n_samples = 100
    n_features = 14

    traces = np.random.randn(n_samples, n_features) * 10 + 30
    traces = np.maximum(traces, 0)

    labels = np.random.randint(0, 2, n_samples)

    temporal_features = extractor.extract_sequences(traces, labels)

    print(f"   Original traces: {traces.shape}")
    print(f"   Sequences: {temporal_features.sequences.shape}")
    print(f"   Lengths: {temporal_features.lengths.shape}")
    print(f"   Labels: {temporal_features.labels.shape}")

    # Test 4: Training step
    print("\n4. Testing Training Step:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Convert to tensors
    x_train = torch.from_numpy(temporal_features.sequences[:batch_size]).float()
    y_train = torch.from_numpy(temporal_features.labels[:batch_size]).long()
    lengths_train = torch.from_numpy(temporal_features.lengths[:batch_size]).long()

    # Forward pass
    logits = model(x_train, lengths_train)
    loss = criterion(logits, y_train)

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
        preds, probs = model.predict(x_train, lengths_train)

    accuracy = (preds == y_train).float().mean()
    print(f"   Predictions: {preds.tolist()}")
    print(f"   Ground truth: {y_train.tolist()}")
    print(f"   Accuracy: {accuracy:.2%}")

    # Test 6: Single trace prediction
    print("\n6. Testing Single Trace Prediction:")
    single_trace = traces[0]  # (n_features,)
    single_seq = extractor.extract_single_sequence(single_trace)

    x_single = torch.from_numpy(single_seq).unsqueeze(0).float()  # (1, 1, n_features)

    with torch.no_grad():
        pred, prob = model.predict(x_single)

    print(f"   Single trace shape: {single_trace.shape}")
    print(f"   Prediction: {pred.item()} (anomaly prob: {prob.item():.3f})")

    print("\n✅ LSTM Anomaly Detector tests passed!")
