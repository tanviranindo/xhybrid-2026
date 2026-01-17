"""
Explainer Module
Provides multiple explanation methods for anomaly detection.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from captum.attr import (
        IntegratedGradients,
        LayerIntegratedGradients,
        DeepLift,
        GradientShap,
        Saliency
    )
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


@dataclass
class Explanation:
    """Container for explanation results."""
    method: str
    trace_id: str
    prediction: float
    prediction_label: str

    # Feature-level explanations
    feature_importance: Dict[str, float]
    component_contributions: Dict[str, float]

    # Detailed explanations
    grammar_activations: Optional[List[Dict]] = None
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    service_importance: Optional[Dict[str, float]] = None

    # Raw data
    raw_shap_values: Optional[np.ndarray] = None


class SHAPExplainer:
    """
    SHAP-based explainer for feature importance.
    Uses KernelSHAP for model-agnostic explanations.
    """

    def __init__(self,
                 model: nn.Module,
                 background_data: Optional[Dict[str, torch.Tensor]] = None,
                 n_background: int = 100):
        """
        Args:
            model: The anomaly detection model
            background_data: Background dataset for SHAP
            n_background: Number of background samples
        """
        if not HAS_SHAP:
            raise ImportError("shap package required. Install with: pip install shap")

        self.model = model
        self.model.eval()
        self.background_data = background_data
        self.n_background = n_background
        self.explainer = None

    def _prepare_input(self, data: Dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten inputs for SHAP."""
        flat_features = []
        self.feature_names = []

        for key in ['sequences', 'latencies', 'grammar_features']:
            if key in data:
                arr = data[key].cpu().numpy()
                if len(arr.shape) > 1:
                    for i in range(arr.shape[1]):
                        flat_features.append(arr[:, i])
                        self.feature_names.append(f"{key}_{i}")
                else:
                    flat_features.append(arr)
                    self.feature_names.append(key)

        return np.column_stack(flat_features)

    def _model_wrapper(self, flat_input: np.ndarray) -> np.ndarray:
        """Wrapper for SHAP to call model."""
        # Reconstruct input tensors
        device = next(self.model.parameters()).device

        # Parse flat input back to structured format
        # This is a simplified version - actual implementation needs
        # to track feature boundaries
        seq_len = 50  # Adjust based on your config
        n_grammar = 100

        n_samples = flat_input.shape[0]
        results = []

        for i in range(n_samples):
            row = flat_input[i]

            sequences = torch.tensor(
                row[:seq_len].astype(np.int64),
                device=device
            ).unsqueeze(0)

            latencies = torch.tensor(
                row[seq_len:2*seq_len].astype(np.float32),
                device=device
            ).unsqueeze(0)

            grammar = torch.tensor(
                row[2*seq_len:2*seq_len+n_grammar].astype(np.float32),
                device=device
            ).unsqueeze(0)

            with torch.no_grad():
                logits = self.model(sequences, latencies, grammar)
                prob = torch.softmax(logits, dim=-1)[0, 1].cpu().numpy()

            results.append(prob)

        return np.array(results)

    def explain(self,
                data: Dict[str, torch.Tensor],
                trace_id: str = "unknown") -> Explanation:
        """
        Generate SHAP explanation for a sample.

        Args:
            data: Input data dictionary
            trace_id: Trace identifier

        Returns:
            Explanation object with SHAP values
        """
        # Prepare data
        flat_input = self._prepare_input(data)

        # Create background if needed
        if self.background_data is not None:
            background = self._prepare_input(self.background_data)
            background = background[:self.n_background]
        else:
            background = flat_input

        # Create explainer
        self.explainer = shap.KernelExplainer(self._model_wrapper, background)

        # Compute SHAP values
        shap_values = self.explainer.shap_values(flat_input)

        # Get prediction
        with torch.no_grad():
            logits = self.model(
                data['sequences'],
                data['latencies'],
                data['grammar_features']
            )
            prob = torch.softmax(logits, dim=-1)[0, 1].item()

        # Aggregate feature importance
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            # Group by feature type
            base_name = name.rsplit('_', 1)[0]
            if base_name not in feature_importance:
                feature_importance[base_name] = 0.0
            feature_importance[base_name] += abs(shap_values[0, i])

        # Normalize
        total = sum(feature_importance.values())
        if total > 0:
            feature_importance = {k: v/total for k, v in feature_importance.items()}

        # Component contributions
        component_contributions = {
            'syntactic': feature_importance.get('grammar_features', 0) + \
                        feature_importance.get('latencies', 0) * 0.5,
            'temporal': feature_importance.get('sequences', 0) + \
                       feature_importance.get('latencies', 0) * 0.5,
            'graph': 0.0  # Graph features not in flat input
        }

        return Explanation(
            method='SHAP',
            trace_id=trace_id,
            prediction=prob,
            prediction_label='anomaly' if prob > 0.5 else 'normal',
            feature_importance=feature_importance,
            component_contributions=component_contributions,
            raw_shap_values=shap_values
        )


class AttentionExplainer:
    """
    Attention-based explainer using model's attention weights.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model with attention mechanisms
        """
        self.model = model
        self.model.eval()

    def explain(self,
                data: Dict[str, torch.Tensor],
                trace_id: str = "unknown",
                service_vocab: Optional[Dict[str, int]] = None) -> Explanation:
        """
        Generate attention-based explanations.

        Args:
            data: Input data dictionary
            trace_id: Trace identifier
            service_vocab: Service name to index mapping

        Returns:
            Explanation with attention weights
        """
        device = next(self.model.parameters()).device

        # Forward pass with attention
        with torch.no_grad():
            # Get temporal attention
            temporal_emb, temporal_attn = self.model.temporal_encoder(
                data['sequences'].to(device),
                data['latencies'].to(device),
                return_attention=True
            )

            # Get graph attention if available
            graph_attn = None
            if hasattr(self.model, 'graph_encoder'):
                if hasattr(self.model.graph_encoder, 'encoder'):
                    if hasattr(self.model.graph_encoder.encoder, 'get_attention_weights'):
                        graph_attn = self.model.graph_encoder.encoder.get_attention_weights()

            # Get full prediction
            logits = self.model(
                data['sequences'].to(device),
                data['latencies'].to(device),
                data['grammar_features'].to(device)
            )
            prob = torch.softmax(logits, dim=-1)[0, 1].item()

        # Process temporal attention
        attention_weights = {}
        if temporal_attn is not None:
            temporal_attn_np = temporal_attn.cpu().numpy()
            attention_weights['temporal'] = temporal_attn_np

            # Map to services if vocab provided
            if service_vocab is not None:
                idx_to_service = {v: k for k, v in service_vocab.items()}
                seq = data['sequences'][0].cpu().numpy()

                service_attention = {}
                for i, idx in enumerate(seq):
                    if idx > 0:  # Skip padding
                        service = idx_to_service.get(idx, f"service_{idx}")
                        if service not in service_attention:
                            service_attention[service] = 0.0
                        service_attention[service] += temporal_attn_np[0, i]

                # Normalize
                total = sum(service_attention.values())
                if total > 0:
                    service_attention = {k: v/total for k, v in service_attention.items()}
            else:
                service_attention = None
        else:
            service_attention = None

        if graph_attn is not None:
            attention_weights['graph'] = graph_attn

        # Component contributions from model
        component_contributions = self.model.get_component_contributions(
            data['sequences'].to(device),
            data['latencies'].to(device),
            data['grammar_features'].to(device)
        )

        return Explanation(
            method='Attention',
            trace_id=trace_id,
            prediction=prob,
            prediction_label='anomaly' if prob > 0.5 else 'normal',
            feature_importance={},
            component_contributions=component_contributions,
            attention_weights=attention_weights,
            service_importance=service_attention
        )


class GrammarExplainer:
    """
    Grammar rule-based explainer.
    Shows which syntactic patterns were activated/violated.
    """

    def __init__(self,
                 grammar_inference,
                 normal_patterns: Optional[Dict[str, float]] = None):
        """
        Args:
            grammar_inference: Fitted GrammarInference object
            normal_patterns: Baseline pattern frequencies for normal traces
        """
        self.grammar = grammar_inference
        self.normal_patterns = normal_patterns or {}

    def explain(self,
                service_sequence: List[str],
                trace_id: str = "unknown",
                prediction: float = 0.0) -> Explanation:
        """
        Generate grammar-based explanations.

        Args:
            service_sequence: List of service names in call order
            trace_id: Trace identifier
            prediction: Model prediction probability

        Returns:
            Explanation with grammar activations
        """
        # Get rule activations
        activations = self.grammar.get_rule_activations(service_sequence)

        grammar_activations = []
        missing_normal = []
        unexpected = []

        for rule, is_active in activations:
            activation_info = {
                'rule_id': rule.lhs,
                'pattern': ' -> '.join(rule.rhs),
                'is_active': is_active,
                'support': rule.support,
                'frequency': rule.frequency
            }

            # Check against normal baseline
            if self.normal_patterns:
                normal_freq = self.normal_patterns.get(rule.lhs, 0.0)
                activation_info['normal_frequency'] = normal_freq

                if is_active and normal_freq < 0.1:
                    activation_info['deviation'] = 'unexpected'
                    unexpected.append(rule.lhs)
                elif not is_active and normal_freq > 0.5:
                    activation_info['deviation'] = 'missing'
                    missing_normal.append(rule.lhs)
                else:
                    activation_info['deviation'] = 'normal'

            grammar_activations.append(activation_info)

        # Sort by support
        grammar_activations.sort(key=lambda x: x['support'], reverse=True)

        # Compute deviation score
        n_deviations = len(missing_normal) + len(unexpected)
        n_rules = len(activations)
        deviation_score = n_deviations / max(n_rules, 1)

        return Explanation(
            method='Grammar',
            trace_id=trace_id,
            prediction=prediction,
            prediction_label='anomaly' if prediction > 0.5 else 'normal',
            feature_importance={
                'missing_patterns': len(missing_normal),
                'unexpected_patterns': len(unexpected),
                'deviation_score': deviation_score
            },
            component_contributions={'syntactic': 1.0},
            grammar_activations=grammar_activations
        )


class ServiceLevelExplainer:
    """
    Service-level explainer for understanding per-service contributions.
    """

    def __init__(self, model: nn.Module, service_vocab: Dict[str, int]):
        """
        Args:
            model: The anomaly detection model
            service_vocab: Service name to index mapping
        """
        self.model = model
        self.service_vocab = service_vocab
        self.idx_to_service = {v: k for k, v in service_vocab.items()}

    def explain(self,
                data: Dict[str, torch.Tensor],
                trace_id: str = "unknown") -> Explanation:
        """
        Generate service-level explanations.

        Args:
            data: Input data dictionary
            trace_id: Trace identifier

        Returns:
            Explanation with service importance
        """
        device = next(self.model.parameters()).device

        sequences = data['sequences'].to(device)
        latencies = data['latencies'].to(device)
        grammar_features = data['grammar_features'].to(device)

        # Baseline prediction
        with torch.no_grad():
            base_logits = self.model(sequences, latencies, grammar_features)
            base_prob = torch.softmax(base_logits, dim=-1)[0, 1].item()

        # Compute service importance via perturbation
        service_importance = {}
        seq_np = sequences[0].cpu().numpy()

        for i, idx in enumerate(seq_np):
            if idx == 0:  # Skip padding
                continue

            service = self.idx_to_service.get(idx, f"service_{idx}")

            # Mask this position
            perturbed_seq = sequences.clone()
            perturbed_seq[0, i] = 0  # Mask with padding

            with torch.no_grad():
                pert_logits = self.model(perturbed_seq, latencies, grammar_features)
                pert_prob = torch.softmax(pert_logits, dim=-1)[0, 1].item()

            # Importance = change in prediction
            importance = abs(base_prob - pert_prob)

            if service not in service_importance:
                service_importance[service] = 0.0
            service_importance[service] = max(service_importance[service], importance)

        # Normalize
        total = sum(service_importance.values())
        if total > 0:
            service_importance = {k: v/total for k, v in service_importance.items()}

        # Sort by importance
        service_importance = dict(
            sorted(service_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return Explanation(
            method='ServiceLevel',
            trace_id=trace_id,
            prediction=base_prob,
            prediction_label='anomaly' if base_prob > 0.5 else 'normal',
            feature_importance={},
            component_contributions={},
            service_importance=service_importance
        )


class HybridExplainer:
    """
    Unified explainer combining multiple explanation methods.
    """

    def __init__(self,
                 model: nn.Module,
                 grammar_inference=None,
                 service_vocab: Optional[Dict[str, int]] = None,
                 background_data: Optional[Dict[str, torch.Tensor]] = None,
                 use_shap: bool = True,
                 use_attention: bool = True,
                 use_grammar: bool = True,
                 use_service: bool = True):
        """
        Args:
            model: The anomaly detection model
            grammar_inference: Fitted GrammarInference object
            service_vocab: Service vocabulary
            background_data: Background data for SHAP
            use_*: Whether to use each explanation method
        """
        self.model = model
        self.service_vocab = service_vocab

        self.explainers = {}

        if use_shap and HAS_SHAP:
            try:
                self.explainers['shap'] = SHAPExplainer(model, background_data)
            except Exception:
                pass

        if use_attention:
            self.explainers['attention'] = AttentionExplainer(model)

        if use_grammar and grammar_inference is not None:
            self.explainers['grammar'] = GrammarExplainer(grammar_inference)

        if use_service and service_vocab is not None:
            self.explainers['service'] = ServiceLevelExplainer(model, service_vocab)

    def explain(self,
                data: Dict[str, torch.Tensor],
                service_sequence: Optional[List[str]] = None,
                trace_id: str = "unknown") -> Dict[str, Explanation]:
        """
        Generate explanations using all available methods.

        Args:
            data: Input data dictionary
            service_sequence: Original service call sequence
            trace_id: Trace identifier

        Returns:
            Dictionary of explanations from each method
        """
        explanations = {}

        # Get base prediction first
        device = next(self.model.parameters()).device
        with torch.no_grad():
            logits = self.model(
                data['sequences'].to(device),
                data['latencies'].to(device),
                data['grammar_features'].to(device)
            )
            base_prob = torch.softmax(logits, dim=-1)[0, 1].item()

        # SHAP explanation
        if 'shap' in self.explainers:
            try:
                explanations['shap'] = self.explainers['shap'].explain(data, trace_id)
            except Exception as e:
                print(f"SHAP explanation failed: {e}")

        # Attention explanation
        if 'attention' in self.explainers:
            explanations['attention'] = self.explainers['attention'].explain(
                data, trace_id, self.service_vocab
            )

        # Grammar explanation
        if 'grammar' in self.explainers and service_sequence is not None:
            explanations['grammar'] = self.explainers['grammar'].explain(
                service_sequence, trace_id, base_prob
            )

        # Service-level explanation
        if 'service' in self.explainers:
            explanations['service'] = self.explainers['service'].explain(
                data, trace_id
            )

        return explanations

    def get_summary(self, explanations: Dict[str, Explanation]) -> Dict[str, Any]:
        """
        Create a summary of all explanations.

        Args:
            explanations: Dictionary of explanations

        Returns:
            Summary dictionary
        """
        if not explanations:
            return {}

        # Get prediction from any explanation
        first_exp = next(iter(explanations.values()))

        summary = {
            'trace_id': first_exp.trace_id,
            'prediction': first_exp.prediction,
            'prediction_label': first_exp.prediction_label,
            'methods_used': list(explanations.keys()),
            'component_contributions': {},
            'top_services': [],
            'grammar_deviations': []
        }

        # Aggregate component contributions
        for exp in explanations.values():
            for comp, val in exp.component_contributions.items():
                if comp not in summary['component_contributions']:
                    summary['component_contributions'][comp] = []
                summary['component_contributions'][comp].append(val)

        # Average component contributions
        for comp in summary['component_contributions']:
            vals = summary['component_contributions'][comp]
            summary['component_contributions'][comp] = sum(vals) / len(vals)

        # Top services
        if 'service' in explanations:
            service_imp = explanations['service'].service_importance or {}
            summary['top_services'] = list(service_imp.items())[:5]

        # Grammar deviations
        if 'grammar' in explanations:
            activations = explanations['grammar'].grammar_activations or []
            for act in activations:
                if act.get('deviation') in ['missing', 'unexpected']:
                    summary['grammar_deviations'].append({
                        'pattern': act['pattern'],
                        'type': act['deviation']
                    })

        return summary
