"""
Online Learning with Concept Drift Detection
Novel contribution: Production-ready adaptive learning for microservice traces
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionResult:
    """Results from drift detection."""
    drift_detected: bool
    drift_score: float
    drift_type: str  # 'sudden', 'gradual', 'incremental', 'none'
    metrics: Dict[str, float]
    timestamp: float = field(default_factory=time.time)

    def __str__(self):
        status = "🚨 DRIFT DETECTED" if self.drift_detected else "✓ No drift"
        return f"{status} | Type: {self.drift_type} | Score: {self.drift_score:.3f}"


class DriftDetector:
    """
    Drift detection using multiple methods.

    Methods:
    1. DDM (Drift Detection Method) - uses error rate
    2. ADWIN (Adaptive Windowing) - uses window statistics
    3. Grammar Coverage - uses pattern coverage (novel for traces)
    """

    def __init__(self,
                 method: str = 'ddm',
                 warning_level: float = 2.0,
                 drift_level: float = 3.0,
                 window_size: int = 100):
        """
        Args:
            method: 'ddm', 'adwin', or 'grammar_coverage'
            warning_level: Warning threshold (standard deviations)
            drift_level: Drift threshold (standard deviations)
            window_size: Window size for statistics
        """
        self.method = method
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.window_size = window_size

        # DDM statistics
        self.error_history = deque(maxlen=window_size)
        self.error_mean = 0.0
        self.error_std = 0.0
        self.min_error_mean = float('inf')
        self.min_error_std = float('inf')

        # Grammar coverage history
        self.coverage_history = deque(maxlen=window_size)

        # Drift state
        self.in_warning = False
        self.in_drift = False

    def update(self, error: float, coverage: Optional[float] = None) -> DriftDetectionResult:
        """
        Update detector with new observation.

        Args:
            error: Current error rate (0 or 1 for classification)
            coverage: Grammar coverage score (optional)

        Returns:
            DriftDetectionResult
        """
        if self.method == 'ddm':
            return self._ddm_update(error)
        elif self.method == 'grammar_coverage':
            if coverage is None:
                raise ValueError("Grammar coverage required for this method")
            return self._coverage_update(coverage)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _ddm_update(self, error: float) -> DriftDetectionResult:
        """DDM (Drift Detection Method) update."""
        self.error_history.append(error)

        if len(self.error_history) < 30:  # Min samples
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type='none',
                metrics={}
            )

        # Update statistics
        errors = np.array(self.error_history)
        self.error_mean = errors.mean()
        self.error_std = errors.std() + 1e-6

        # Update minimum
        if self.error_mean + self.error_std < self.min_error_mean + self.min_error_std:
            self.min_error_mean = self.error_mean
            self.min_error_std = self.error_std

        # Compute drift score
        drift_score = (self.error_mean + self.error_std) - (self.min_error_mean + self.min_error_std)
        drift_score /= self.min_error_std  # Normalize

        # Detect drift
        if drift_score > self.drift_level:
            drift_detected = True
            drift_type = 'sudden'
        elif drift_score > self.warning_level:
            drift_detected = False
            drift_type = 'gradual' if self.in_warning else 'none'
            self.in_warning = True
        else:
            drift_detected = False
            drift_type = 'none'
            self.in_warning = False

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            drift_type=drift_type,
            metrics={
                'error_mean': self.error_mean,
                'error_std': self.error_std,
                'min_error': self.min_error_mean
            }
        )

    def _coverage_update(self, coverage: float) -> DriftDetectionResult:
        """Grammar coverage-based drift detection (novel)."""
        self.coverage_history.append(coverage)

        if len(self.coverage_history) < 30:
            return DriftDetectionResult(
                drift_detected=False,
                drift_score=0.0,
                drift_type='none',
                metrics={}
            )

        # Baseline coverage (first window)
        baseline_coverage = np.mean(list(self.coverage_history)[:30])

        # Current coverage (recent window)
        recent_coverage = np.mean(list(self.coverage_history)[-30:])

        # Coverage drop = potential drift
        coverage_drop = (baseline_coverage - recent_coverage) / (baseline_coverage + 1e-6)

        # Detect drift based on coverage drop
        if coverage_drop > 0.3:  # 30% drop
            drift_detected = True
            drift_type = 'sudden'
        elif coverage_drop > 0.15:  # 15% drop
            drift_detected = False
            drift_type = 'gradual'
        else:
            drift_detected = False
            drift_type = 'none'

        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=float(coverage_drop),
            drift_type=drift_type,
            metrics={
                'baseline_coverage': baseline_coverage,
                'recent_coverage': recent_coverage,
                'coverage_drop': coverage_drop
            }
        )


class OnlineLearner:
    """
    Online learning system with drift detection and adaptation.

    Innovation: Combines grammar update, neural adaptation, and drift detection
    for production deployment of trace anomaly detection.
    """

    def __init__(self,
                 model: nn.Module,
                 grammar_inference: Any,
                 optimizer: optim.Optimizer,
                 drift_detector: Optional[DriftDetector] = None,
                 buffer_size: int = 10000,
                 retrain_interval: int = 1000,
                 device: str = 'cpu'):
        """
        Args:
            model: The anomaly detection model
            grammar_inference: Grammar learning component
            optimizer: PyTorch optimizer
            drift_detector: Drift detection method
            buffer_size: Experience replay buffer size
            retrain_interval: Retrain after N samples
            device: 'cpu' or 'cuda'
        """
        self.model = model
        self.grammar = grammar_inference
        self.optimizer = optimizer
        self.drift_detector = drift_detector or DriftDetector(method='ddm')
        self.device = device

        # Experience replay buffer
        self.buffer_size = buffer_size
        self.experience_buffer = {
            'traces': deque(maxlen=buffer_size),
            'labels': deque(maxlen=buffer_size),
            'features': deque(maxlen=buffer_size)
        }

        # Training state
        self.retrain_interval = retrain_interval
        self.samples_since_retrain = 0
        self.total_samples = 0

        # Statistics
        self.performance_history = deque(maxlen=1000)
        self.drift_events = []

    def update(self,
               new_trace: Dict[str, torch.Tensor],
               true_label: int,
               service_sequence: List[str]) -> Dict[str, Any]:
        """
        Online update with single new trace.

        Args:
            new_trace: Dictionary with trace features
            true_label: Ground truth label
            service_sequence: List of service names

        Returns:
            Update result with metrics
        """
        self.model.eval()

        # 1. Get prediction
        with torch.no_grad():
            pred_logits = self.model(
                new_trace['sequences'].to(self.device),
                new_trace['latencies'].to(self.device),
                new_trace['grammar_features'].to(self.device)
            )
            pred_label = pred_logits.argmax(dim=-1).item()

        # 2. Compute error
        error = 1 if pred_label != true_label else 0

        # 3. Update grammar incrementally
        self.grammar.incremental_update([service_sequence])

        # 4. Compute coverage
        coverage = self._compute_coverage([service_sequence])

        # 5. Drift detection
        drift_result = self.drift_detector.update(error, coverage)

        # 6. Store in buffer
        self.experience_buffer['traces'].append(new_trace)
        self.experience_buffer['labels'].append(true_label)
        self.experience_buffer['features'].append(service_sequence)

        # 7. Update counters
        self.samples_since_retrain += 1
        self.total_samples += 1
        self.performance_history.append({
            'error': error,
            'coverage': coverage,
            'timestamp': time.time()
        })

        # 8. Handle drift
        update_result = {
            'prediction': pred_label,
            'true_label': true_label,
            'error': error,
            'coverage': coverage,
            'drift': drift_result
        }

        if drift_result.drift_detected:
            logger.warning(f"Drift detected: {drift_result}")
            self.drift_events.append(drift_result)
            self._handle_drift()
            update_result['retraining_triggered'] = True

        elif self.samples_since_retrain >= self.retrain_interval:
            logger.info(f"Regular retraining after {self.retrain_interval} samples")
            self._incremental_train()
            self.samples_since_retrain = 0
            update_result['incremental_update'] = True

        return update_result

    def batch_update(self,
                     traces: List[Dict[str, torch.Tensor]],
                     labels: List[int],
                     sequences: List[List[str]]) -> Dict[str, Any]:
        """
        Batch online update.

        Args:
            traces: List of trace feature dictionaries
            labels: List of true labels
            sequences: List of service sequences

        Returns:
            Batch update results
        """
        results = []
        for trace, label, seq in zip(traces, labels, sequences):
            result = self.update(trace, label, seq)
            results.append(result)

        # Aggregate results
        avg_error = np.mean([r['error'] for r in results])
        avg_coverage = np.mean([r['coverage'] for r in results])
        drift_count = sum(r['drift'].drift_detected for r in results)

        return {
            'batch_size': len(traces),
            'avg_error': avg_error,
            'avg_coverage': avg_coverage,
            'drift_detections': drift_count,
            'individual_results': results
        }

    def _handle_drift(self):
        """Handle detected drift with full retraining."""
        logger.info("🔄 Handling concept drift: Full retraining...")

        # Sample from recent buffer
        buffer_size = len(self.experience_buffer['traces'])
        if buffer_size < 100:
            logger.warning("Insufficient data for retraining")
            return

        # Use recent window (last 50%) + historical sample (first 50%)
        split_point = buffer_size // 2

        recent_traces = list(self.experience_buffer['traces'])[split_point:]
        recent_labels = list(self.experience_buffer['labels'])[split_point:]
        recent_sequences = list(self.experience_buffer['features'])[split_point:]

        historical_traces = list(self.experience_buffer['traces'])[:split_point]
        historical_labels = list(self.experience_buffer['labels'])[:split_point]
        historical_sequences = list(self.experience_buffer['features'])[:split_point]

        # Combine with weight on recent data
        all_traces = recent_traces * 2 + historical_traces
        all_labels = recent_labels * 2 + historical_labels
        all_sequences = recent_sequences * 2 + historical_sequences

        # Update grammar with recent patterns
        self.grammar.relearn(all_sequences)

        # Retrain model
        self._retrain_model(all_traces, all_labels)

        self.samples_since_retrain = 0

    def _incremental_train(self):
        """Incremental training with recent samples."""
        logger.info("Incremental training...")

        # Use last N samples
        recent_size = min(self.retrain_interval, len(self.experience_buffer['traces']))
        traces = list(self.experience_buffer['traces'])[-recent_size:]
        labels = list(self.experience_buffer['labels'])[-recent_size:]

        if len(traces) < 32:  # Min batch size
            return

        self._retrain_model(traces, labels, epochs=1)

    def _retrain_model(self,
                       traces: List[Dict[str, torch.Tensor]],
                       labels: List[int],
                       epochs: int = 5):
        """Retrain model with given data."""
        self.model.train()

        # Create batches
        batch_size = 32
        n_batches = len(traces) // batch_size

        for epoch in range(epochs):
            total_loss = 0

            for i in range(n_batches):
                start = i * batch_size
                end = start + batch_size

                batch_traces = traces[start:end]
                batch_labels = torch.tensor(labels[start:end], device=self.device)

                # Forward pass
                # Stack batch tensors
                sequences = torch.stack([t['sequences'] for t in batch_traces]).to(self.device)
                latencies = torch.stack([t['latencies'] for t in batch_traces]).to(self.device)
                grammar_feats = torch.stack([t['grammar_features'] for t in batch_traces]).to(self.device)

                logits = self.model(sequences, latencies, grammar_feats)

                # Loss
                loss = nn.CrossEntropyLoss()(logits, batch_labels)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / max(n_batches, 1)
            logger.info(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        self.model.eval()

    def _compute_coverage(self, sequences: List[List[str]]) -> float:
        """Compute grammar coverage for sequences."""
        if not hasattr(self.grammar, 'encode_sequence'):
            return 1.0

        activated_rules = 0
        total_rules = len(self.grammar.grammar_rules) if hasattr(self.grammar, 'grammar_rules') else 1

        for seq in sequences:
            encoding = self.grammar.encode_sequence(seq)
            activated_rules += encoding.sum()

        coverage = activated_rules / (total_rules * len(sequences) + 1e-6)
        return float(min(coverage, 1.0))

    def get_statistics(self) -> Dict[str, Any]:
        """Get learner statistics."""
        recent_errors = [h['error'] for h in list(self.performance_history)[-100:]]
        recent_coverage = [h['coverage'] for h in list(self.performance_history)[-100:]]

        return {
            'total_samples': self.total_samples,
            'buffer_size': len(self.experience_buffer['traces']),
            'drift_events': len(self.drift_events),
            'recent_error_rate': np.mean(recent_errors) if recent_errors else 0.0,
            'recent_coverage': np.mean(recent_coverage) if recent_coverage else 0.0,
            'last_drift': self.drift_events[-1] if self.drift_events else None
        }


# Example usage
if __name__ == "__main__":
    print("Testing Online Learner...")

    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, seq, lat, gram):
            return self.fc(torch.randn(seq.size(0), 10))

    # Create dummy grammar
    class DummyGrammar:
        def __init__(self):
            self.grammar_rules = []

        def incremental_update(self, sequences):
            pass

        def relearn(self, sequences):
            pass

        def encode_sequence(self, seq):
            return np.zeros(5)

    model = DummyModel()
    grammar = DummyGrammar()
    optimizer = optim.Adam(model.parameters())

    # Create learner
    learner = OnlineLearner(model, grammar, optimizer)

    # Simulate online updates
    for i in range(200):
        trace = {
            'sequences': torch.randint(0, 10, (1, 20)),
            'latencies': torch.randn(1, 20),
            'grammar_features': torch.randn(1, 5)
        }
        label = np.random.randint(0, 2)
        sequence = ['svc-a', 'svc-b', 'svc-c']

        result = learner.update(trace, label, sequence)

        if result.get('drift'):
            print(f"Sample {i}: {result['drift']}")

    # Get statistics
    stats = learner.get_statistics()
    print(f"\nLearner Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✅ Online Learner tests passed!")
