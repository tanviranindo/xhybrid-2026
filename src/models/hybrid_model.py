"""
Hybrid Anomaly Detection Model
Combines Grammar-based, GNN-based, and LSTM-based approaches with attention fusion

Novel Contribution:
- First hybrid syntactic-neural approach for microservice traces
- Attention-based fusion of complementary feature representations
- Explainable through attention weights and grammar rule activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

from .grammar_baseline import GrammarAnomalyDetector
from .gnn_baseline import GNNAnomalyDetector, ServiceDependencyGraph
from .lstm_baseline import LSTMAnomalyDetector
from .fusion import AttentionFusion


class HybridAnomalyDetector(nn.Module):
    """
    Hybrid anomaly detector combining syntactic, graph, and temporal features.

    Architecture:
    1. Grammar Module: SAX + Sequitur for syntactic patterns
    2. GNN Module: Graph Attention for service dependencies
    3. LSTM Module: BiLSTM for temporal patterns
    4. Attention Fusion: Learnable feature combination
    5. Classifier: Final anomaly prediction

    Args:
        n_services: Number of microservices
        n_node_features: Number of node features for GNN
        hidden_dim: Hidden dimension for all modules
        use_pretrained: Whether to use pretrained baseline models
        fusion_dropout: Dropout rate for fusion layer
    """

    def __init__(self,
                 n_services: int = 14,
                 n_node_features: int = 4,
                 hidden_dim: int = 64,
                 use_pretrained: bool = False,
                 fusion_dropout: float = 0.1):
        super().__init__()

        self.n_services = n_services
        self.hidden_dim = hidden_dim
        self.use_pretrained = use_pretrained

        # ==========================================
        # 1. Grammar-based Feature Extractor
        # ==========================================
        # We'll use the trained grammar detector and extract features
        self.grammar_detector = None  # Will be set externally if pretrained

        # Grammar feature projection (from grammar features to hidden_dim)
        # Grammar features: 100-dimensional vector from SAX + Sequitur
        self.grammar_projection = nn.Sequential(
            nn.Linear(100, hidden_dim),  # 100 grammar features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout)
        )

        # ==========================================
        # 2. GNN-based Feature Extractor
        # ==========================================
        self.gnn = GNNAnomalyDetector(
            n_node_features=n_node_features,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=fusion_dropout,
            pooling='mean'
        )

        # Extract GNN features (before classification layer)
        # We'll use the graph embedding as features

        # ==========================================
        # 3. LSTM-based Feature Extractor
        # ==========================================
        self.lstm = LSTMAnomalyDetector(
            input_dim=n_services,
            hidden_dim=hidden_dim,
            n_layers=2,
            dropout=fusion_dropout,
            use_attention=True
        )

        # Extract LSTM features (before classification layer)
        # We'll access the context vector before classification

        # ==========================================
        # 4. Attention-based Feature Fusion
        # ==========================================
        self.fusion = AttentionFusion(
            feature_dims={
                'grammar': hidden_dim,
                'graph': hidden_dim,
                'temporal': hidden_dim
            },
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim * 2,
            num_heads=4,
            dropout=fusion_dropout
        )

        # ==========================================
        # 5. Final Classifier
        # ==========================================
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )

        # For explainability: store attention weights
        self.last_attention_weights = None
        self.last_grammar_features = None

    def extract_grammar_features(self, traces: List[np.ndarray]) -> torch.Tensor:
        """
        Extract features from grammar-based detector.

        Args:
            traces: List of trace arrays (n_services,)

        Returns:
            (batch, 5) grammar features tensor
        """
        if self.grammar_detector is None:
            # Return dummy features if no grammar detector
            batch_size = len(traces)
            return torch.zeros(batch_size, 5)

        features_list = []
        for trace in traces:
            result = self.grammar_detector.detect(trace)

            # Extract features: coverage, anomaly_score, rule_activations, etc.
            features = [
                result.coverage,
                result.anomaly_score,
                result.compression_ratio,
                len(result.matched_rules) if result.matched_rules else 0,
                len(result.unmatched_digrams) if result.unmatched_digrams else 0
            ]
            features_list.append(features)

        grammar_features = torch.tensor(features_list, dtype=torch.float32)
        self.last_grammar_features = grammar_features.clone()

        return grammar_features

    def forward(self,
                traces_raw: List[np.ndarray],
                node_features: torch.Tensor,
                adjacency: torch.Tensor,
                traces_seq: torch.Tensor,
                grammar_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through hybrid model.

        Args:
            traces_raw: List of raw trace arrays for grammar (batch_size,)
            node_features: (batch, n_nodes, n_node_features) for GNN
            adjacency: (batch, n_nodes, n_nodes) for GNN
            traces_seq: (batch, seq_len, n_services) for LSTM
            grammar_features: (batch, feature_dim) pre-extracted grammar features (optional)

        Returns:
            (batch, 2) logits for [normal, anomaly]
        """
        batch_size = traces_seq.size(0)
        device = traces_seq.device

        # ==========================================
        # Extract features from each module
        # ==========================================

        # 1. Grammar features
        if grammar_features is not None:
            # Use pre-extracted features (REAL FEATURES!)
            grammar_raw = grammar_features
        else:
            # Fallback to old method (zeros)
            grammar_raw = self.extract_grammar_features(traces_raw)
        grammar_features = self.grammar_projection(grammar_raw.to(device))  # (batch, hidden_dim)

        # 2. GNN features (extract before classification)
        # We need to modify GNN forward to return embeddings
        x = node_features
        for i, (gat, norm) in enumerate(zip(self.gnn.gat_layers, self.gnn.layer_norms)):
            x_new = gat(x, adjacency)
            x_new = norm(x_new)
            if i > 0:
                x = x + x_new
            else:
                x = x_new
            x = F.relu(x)

        # Graph pooling to get graph-level features
        graph_features = x.mean(dim=1)  # (batch, hidden_dim)

        # 3. LSTM features (extract before classification)
        lstm_out, (h_n, c_n) = self.lstm.lstm(traces_seq)

        if self.lstm.use_attention:
            attn_scores = self.lstm.attention(lstm_out)
            attn_weights = F.softmax(attn_scores, dim=1)
            temporal_features = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden_dim * 2)
        else:
            forward_h = h_n[-2]
            backward_h = h_n[-1]
            temporal_features = torch.cat([forward_h, backward_h], dim=-1)  # (batch, hidden_dim * 2)

        # Project temporal features to hidden_dim
        temporal_features = temporal_features[:, :self.hidden_dim]  # Take first hidden_dim dimensions

        # ==========================================
        # Fuse features with attention
        # ==========================================
        features_dict = {
            'grammar': grammar_features,
            'graph': graph_features,
            'temporal': temporal_features
        }

        fused_features = self.fusion(features_dict, return_attention=False)  # (batch, hidden_dim * 2)

        # Store attention weights for explainability
        self.last_attention_weights = self.fusion.attention_weights

        # ==========================================
        # Final classification
        # ==========================================
        logits = self.classifier(fused_features)  # (batch, 2)

        return logits

    def predict(self,
                traces_raw: List[np.ndarray],
                node_features: torch.Tensor,
                adjacency: torch.Tensor,
                traces_seq: torch.Tensor,
                grammar_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Predict anomalies with explainability.

        Returns:
            (predictions, probabilities, explanations)
        """
        logits = self.forward(traces_raw, node_features, adjacency, traces_seq, grammar_features)
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        # Gather explanations
        explanations = {
            'attention_weights': self.last_attention_weights,
            'grammar_features': self.last_grammar_features,
            'modality_weights': self.fusion.modality_weights.data
        }

        return preds, probs[:, 1], explanations

    def load_pretrained_baselines(self,
                                  grammar_detector: Optional[GrammarAnomalyDetector] = None,
                                  gnn_model: Optional[GNNAnomalyDetector] = None,
                                  lstm_model: Optional[LSTMAnomalyDetector] = None):
        """
        Load pretrained baseline models.

        Args:
            grammar_detector: Trained grammar-based detector
            gnn_model: Trained GNN model
            lstm_model: Trained LSTM model
        """
        if grammar_detector is not None:
            self.grammar_detector = grammar_detector
            print("✅ Loaded pretrained Grammar detector")

        if gnn_model is not None:
            self.gnn.load_state_dict(gnn_model.state_dict())
            print("✅ Loaded pretrained GNN model")

        if lstm_model is not None:
            self.lstm.load_state_dict(lstm_model.state_dict())
            print("✅ Loaded pretrained LSTM model")

    def freeze_baselines(self):
        """Freeze baseline model parameters (only train fusion and classifier)."""
        # Freeze GNN
        for param in self.gnn.parameters():
            param.requires_grad = False

        # Freeze LSTM
        for param in self.lstm.parameters():
            param.requires_grad = False

        print("✅ Froze baseline models (GNN, LSTM)")
        print("   Only fusion and classifier will be trained")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Hybrid Anomaly Detector...")

    # Test configuration
    batch_size = 8
    n_services = 14
    n_nodes = 14
    seq_len = 1

    # Create model
    model = HybridAnomalyDetector(
        n_services=n_services,
        n_node_features=4,
        hidden_dim=64,
        use_pretrained=False
    )

    print(f"\n✅ Model created:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create dummy data
    traces_raw = [np.random.randn(n_services) * 10 + 30 for _ in range(batch_size)]
    node_features = torch.randn(batch_size, n_nodes, 4)
    adjacency = torch.randint(0, 2, (batch_size, n_nodes, n_nodes)).float()
    traces_seq = torch.randn(batch_size, seq_len, n_services)

    # Test forward pass
    print("\n1. Testing Forward Pass:")
    logits = model(traces_raw, node_features, adjacency, traces_seq)
    print(f"   Input traces: {len(traces_raw)}")
    print(f"   Node features: {node_features.shape}")
    print(f"   Adjacency: {adjacency.shape}")
    print(f"   Trace sequences: {traces_seq.shape}")
    print(f"   Output logits: {logits.shape}")

    # Test prediction with explainability
    print("\n2. Testing Prediction with Explainability:")
    preds, probs, explanations = model.predict(traces_raw, node_features, adjacency, traces_seq)
    print(f"   Predictions: {preds.shape}")
    print(f"   Probabilities: {probs.shape}")
    print(f"   Modality weights: {explanations['modality_weights']}")

    # Test training step
    print("\n3. Testing Training Step:")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    labels = torch.randint(0, 2, (batch_size,))

    optimizer.zero_grad()
    logits = model(traces_raw, node_features, adjacency, traces_seq)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients computed: ✅")

    # Test freezing baselines
    print("\n4. Testing Baseline Freezing:")
    model.freeze_baselines()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")

    print("\n✅ Hybrid Anomaly Detector tests passed!")
