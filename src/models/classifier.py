"""
Classifier Module
Final anomaly detection classifier combining all modules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .syntactic import SyntacticEncoder
from .graph import GraphEncoder
from .temporal import TemporalEncoder
from .fusion import HybridFusion, AttentionFusion, AdaptiveFusion


class AnomalyClassifier(nn.Module):
    """
    Binary/Multi-class anomaly classifier.
    Takes fused embeddings and outputs predictions.
    """

    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 128,
                 num_classes: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of output classes (2 for binary)
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, input_dim) fused embeddings

        Returns:
            (batch_size, num_classes) logits
        """
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        if self.num_classes == 2:
            return F.softmax(logits, dim=-1)[:, 1]  # Probability of anomaly
        return F.softmax(logits, dim=-1)


class HybridAnomalyDetector(nn.Module):
    """
    Complete hybrid anomaly detection model.
    Integrates syntactic, graph, and temporal modules with fusion and classification.
    """

    def __init__(self,
                 vocab_size: int = 100,
                 max_seq_len: int = 50,
                 num_grammar_rules: int = 100,
                 # Syntactic module params
                 sax_dim: int = 64,
                 grammar_dim: int = 128,
                 syntactic_output_dim: int = 128,
                 # Graph module params
                 graph_in_channels: int = 6,
                 graph_hidden_dim: int = 64,
                 graph_output_dim: int = 128,
                 graph_architecture: str = 'gat',
                 # Temporal module params
                 temporal_embedding_dim: int = 64,
                 temporal_hidden_dim: int = 128,
                 temporal_output_dim: int = 128,
                 temporal_architecture: str = 'lstm',
                 # Fusion params
                 fusion_hidden_dim: int = 256,
                 fusion_output_dim: int = 256,
                 fusion_type: str = 'hybrid',
                 # Classification params
                 num_classes: int = 2,
                 dropout: float = 0.1):
        """
        Complete model configuration.
        """
        super().__init__()

        # Syntactic encoder
        self.syntactic_encoder = SyntacticEncoder(
            sax_dim=sax_dim,
            grammar_dim=grammar_dim,
            hidden_dim=syntactic_output_dim * 2,
            output_dim=syntactic_output_dim,
            num_grammar_rules=num_grammar_rules,
            dropout=dropout
        )

        # Graph encoder
        self.graph_encoder = GraphEncoder(
            in_channels=graph_in_channels,
            hidden_channels=graph_hidden_dim,
            out_channels=graph_output_dim,
            architecture=graph_architecture,
            dropout=dropout
        )

        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            vocab_size=vocab_size,
            embedding_dim=temporal_embedding_dim,
            hidden_dim=temporal_hidden_dim,
            output_dim=temporal_output_dim,
            architecture=temporal_architecture,
            use_latency=True,
            max_seq_len=max_seq_len,
            dropout=dropout
        )

        # Fusion module
        feature_dims = {
            'syntactic': syntactic_output_dim,
            'graph': graph_output_dim,
            'temporal': temporal_output_dim
        }

        if fusion_type == 'hybrid':
            self.fusion = HybridFusion(
                syntactic_dim=syntactic_output_dim,
                graph_dim=graph_output_dim,
                temporal_dim=temporal_output_dim,
                hidden_dim=fusion_hidden_dim,
                output_dim=fusion_output_dim,
                dropout=dropout
            )
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(
                feature_dims=feature_dims,
                hidden_dim=fusion_hidden_dim,
                output_dim=fusion_output_dim,
                dropout=dropout
            )
        elif fusion_type == 'adaptive':
            self.fusion = AdaptiveFusion(
                feature_dims=feature_dims,
                hidden_dim=fusion_hidden_dim,
                output_dim=fusion_output_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.fusion_type = fusion_type

        # Classifier
        self.classifier = AnomalyClassifier(
            input_dim=fusion_output_dim,
            hidden_dim=fusion_hidden_dim // 2,
            num_classes=num_classes,
            dropout=dropout
        )

        # Store for explainability
        self.last_embeddings = {}

    def forward(self,
                sequences: torch.Tensor,
                latencies: torch.Tensor,
                grammar_features: torch.Tensor,
                graph_x: Optional[torch.Tensor] = None,
                graph_edge_index: Optional[torch.Tensor] = None,
                graph_edge_attr: Optional[torch.Tensor] = None,
                graph_batch: Optional[torch.Tensor] = None,
                return_embeddings: bool = False) -> torch.Tensor:
        """
        Forward pass.

        Args:
            sequences: (batch_size, seq_len) service call indices
            latencies: (batch_size, seq_len) latency values
            grammar_features: (batch_size, num_rules) grammar activations
            graph_x: Node features for graph
            graph_edge_index: Edge connectivity
            graph_edge_attr: Edge features
            graph_batch: Batch assignment for graph nodes
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            (batch_size, num_classes) logits
        """
        batch_size = sequences.size(0)

        # Syntactic encoding
        syntactic_emb = self.syntactic_encoder(latencies, grammar_features)
        self.last_embeddings['syntactic'] = syntactic_emb.detach()

        # Temporal encoding
        temporal_emb = self.temporal_encoder(sequences, latencies)
        self.last_embeddings['temporal'] = temporal_emb.detach()

        # Graph encoding
        if graph_x is not None and graph_edge_index is not None:
            graph_emb = self.graph_encoder(
                graph_x, graph_edge_index,
                edge_attr=graph_edge_attr,
                batch=graph_batch
            )
        else:
            # Fallback: use zeros if no graph provided
            graph_emb = torch.zeros(
                batch_size,
                self.graph_encoder.output_dim,
                device=sequences.device
            )
        self.last_embeddings['graph'] = graph_emb.detach()

        # Fusion
        if self.fusion_type == 'hybrid':
            fused = self.fusion(syntactic_emb, graph_emb, temporal_emb)
        else:
            features = {
                'syntactic': syntactic_emb,
                'graph': graph_emb,
                'temporal': temporal_emb
            }
            fused = self.fusion(features)

        self.last_embeddings['fused'] = fused.detach()

        # Classification
        logits = self.classifier(fused)

        if return_embeddings:
            return logits, self.last_embeddings

        return logits

    def predict(self,
                sequences: torch.Tensor,
                latencies: torch.Tensor,
                grammar_features: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        Get binary predictions.
        """
        logits = self.forward(sequences, latencies, grammar_features, **kwargs)
        return torch.argmax(logits, dim=-1)

    def predict_proba(self,
                      sequences: torch.Tensor,
                      latencies: torch.Tensor,
                      grammar_features: torch.Tensor,
                      **kwargs) -> torch.Tensor:
        """
        Get probability of anomaly.
        """
        logits = self.forward(sequences, latencies, grammar_features, **kwargs)
        return F.softmax(logits, dim=-1)[:, 1]

    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get last computed embeddings for each component."""
        return self.last_embeddings

    def get_component_contributions(self,
                                     sequences: torch.Tensor,
                                     latencies: torch.Tensor,
                                     grammar_features: torch.Tensor,
                                     **kwargs) -> Dict[str, float]:
        """
        Compute contribution of each component to final prediction.
        Useful for explainability.
        """
        _ = self.forward(sequences, latencies, grammar_features, **kwargs)

        contributions = {}
        for name, emb in self.last_embeddings.items():
            if name != 'fused':
                contributions[name] = torch.norm(emb, dim=-1).mean().item()

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v / total for k, v in contributions.items()}

        return contributions


class EnsembleAnomalyDetector(nn.Module):
    """
    Ensemble of multiple anomaly detectors.
    Combines predictions from different model configurations.
    """

    def __init__(self,
                 models: List[nn.Module],
                 combine_method: str = 'average',
                 learnable_weights: bool = False):
        """
        Args:
            models: List of HybridAnomalyDetector models
            combine_method: 'average', 'max', or 'weighted'
            learnable_weights: Whether to learn combination weights
        """
        super().__init__()

        self.models = nn.ModuleList(models)
        self.combine_method = combine_method

        if learnable_weights:
            self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        else:
            self.register_buffer('weights', torch.ones(len(models)) / len(models))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Ensemble forward pass.
        """
        predictions = []
        for model in self.models:
            pred = model(*args, **kwargs)
            predictions.append(pred)

        preds = torch.stack(predictions, dim=0)  # (num_models, batch, num_classes)

        if self.combine_method == 'average':
            return preds.mean(dim=0)
        elif self.combine_method == 'max':
            return preds.max(dim=0)[0]
        elif self.combine_method == 'weighted':
            weights = F.softmax(self.weights, dim=0)
            return (preds * weights.view(-1, 1, 1)).sum(dim=0)
        else:
            raise ValueError(f"Unknown combine method: {self.combine_method}")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for representation learning.
    Encourages similar traces to have similar embeddings.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self,
                embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings: (batch_size, dim) normalized embeddings
            labels: (batch_size,) class labels

        Returns:
            Scalar loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create positive mask
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        positive_mask.fill_diagonal_(0)  # Exclude self

        # Create negative mask
        negative_mask = 1 - positive_mask
        negative_mask.fill_diagonal_(0)

        # Compute loss
        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * positive_mask).sum(dim=-1)
        neg_sum = (exp_sim * negative_mask).sum(dim=-1)

        # Avoid division by zero
        pos_sum = torch.clamp(pos_sum, min=1e-8)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Down-weights easy examples, focuses on hard ones.
    """

    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor for positive class
            gamma: Focusing parameter
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: (batch_size, num_classes) model outputs
            targets: (batch_size,) ground truth labels

        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)

        # Apply focal weighting
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_weight = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha, dtype=torch.float),
            torch.full_like(targets, 1 - self.alpha, dtype=torch.float)
        )

        loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
