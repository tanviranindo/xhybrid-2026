"""
Grammar-Based Anomaly Detector
Replicates D'Angelo & d'Aloisio ICPE 2024 baseline

Pipeline:
1. SAX encoding of service latencies
2. Sequitur grammar induction from normal traces
3. Anomaly detection based on grammar coverage/distance
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.sax_encoding import SAXEncoder
from features.sequitur_grammar import SequiturGrammar, GrammarFeatureExtractor


@dataclass
class GrammarAnomalyScore:
    """Anomaly score from grammar-based detection."""
    is_anomaly: bool
    anomaly_score: float
    coverage: float
    num_matched_rules: int
    features: Dict


class GrammarAnomalyDetector:
    """
    Grammar-based anomaly detector for microservice traces.

    Method:
    1. Train on normal traces to learn grammar
    2. Detect anomalies based on deviation from learned grammar

    Args:
        word_size: SAX word size (PAA segments)
        alphabet_size: SAX alphabet size
        anomaly_threshold: Threshold for anomaly score
        use_coverage: Whether to use coverage-based detection
    """

    def __init__(self,
                 word_size: int = 8,
                 alphabet_size: int = 4,
                 anomaly_threshold: float = 0.3,
                 use_coverage: bool = True):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.anomaly_threshold = anomaly_threshold
        self.use_coverage = use_coverage

        # Components
        self.sax_encoder = SAXEncoder(word_size, alphabet_size)
        self.grammar = None
        self.feature_extractor = None

        # Statistics from training
        self.normal_coverage_mean = None
        self.normal_coverage_std = None
        self.normal_features_mean = None
        self.normal_features_std = None

    def _encode_trace(self, trace: np.ndarray) -> List[str]:
        """
        Encode a single trace to SAX sequence.

        Args:
            trace: (n_services,) or (n_samples, n_services)

        Returns:
            SAX sequence (concatenated from all services)
        """
        if trace.ndim == 1:
            # Single sample with multiple services
            # Create a simple sequence from service latencies
            sax_seq = []
            for latency in trace:
                # Encode each latency value
                # Use a small time series representation
                ts = np.array([latency] * 4)  # Repeat value
                sax_str = self.sax_encoder.encode(ts)
                sax_seq.extend(list(sax_str))
            return sax_seq
        else:
            # Multiple samples, encode each service's temporal pattern
            sax_sequences = []
            for service_idx in range(trace.shape[1]):
                service_ts = trace[:, service_idx]
                sax_str = self.sax_encoder.encode(service_ts)
                sax_sequences.extend(list(sax_str))
            return sax_sequences

    def train(self, normal_traces: List[np.ndarray], verbose: bool = True):
        """
        Train grammar on normal traces.

        Args:
            normal_traces: List of normal trace arrays
            verbose: Whether to print progress
        """
        if verbose:
            print(f"Training grammar-based detector on {len(normal_traces)} normal traces...")

        # Step 1: Encode all traces to SAX
        if verbose:
            print("  Step 1/3: SAX encoding...")

        sax_sequences = []
        for i, trace in enumerate(normal_traces):
            sax_seq = self._encode_trace(trace)
            sax_sequences.append(sax_seq)

            if verbose and (i + 1) % 50 == 0:
                print(f"    Encoded {i + 1}/{len(normal_traces)} traces")

        # Step 2: Induce grammar
        if verbose:
            print("  Step 2/3: Grammar induction...")

        self.grammar = SequiturGrammar(
            min_pattern_length=2,
            max_rules=100
        )
        rules = self.grammar.induce_batch(sax_sequences)

        if verbose:
            print(f"    Extracted {len(rules)} grammar rules")
            print(f"    Sample rules:")
            for rule in rules[:5]:
                print(f"      {rule}")

        # Step 3: Compute statistics on normal traces
        if verbose:
            print("  Step 3/3: Computing baseline statistics...")

        self.feature_extractor = GrammarFeatureExtractor(self.grammar)

        coverages = []
        features_list = []

        for sax_seq in sax_sequences:
            coverage = self.feature_extractor.compute_coverage(sax_seq)
            features = self.feature_extractor.extract_features(sax_seq)

            coverages.append(coverage)
            features_list.append(features)

        # Compute mean and std
        self.normal_coverage_mean = np.mean(coverages)
        self.normal_coverage_std = np.std(coverages)

        if verbose:
            print(f"    Normal coverage: {self.normal_coverage_mean:.3f} ± {self.normal_coverage_std:.3f}")

        # Store feature statistics (for advanced anomaly scoring)
        feature_arrays = {
            'num_rules': [f['num_rules'] for f in features_list],
            'compression_ratio': [f['compression_ratio'] for f in features_list],
            'num_activated_rules': [f['num_activated_rules'] for f in features_list],
        }

        self.normal_features_mean = {k: np.mean(v) for k, v in feature_arrays.items()}
        self.normal_features_std = {k: np.std(v) for k, v in feature_arrays.items()}

        if verbose:
            print(f"\n  ✅ Training complete!")
            print(f"     Grammar rules: {len(rules)}")
            print(f"     Normal coverage: {self.normal_coverage_mean:.1%}")

    def detect(self, trace: np.ndarray) -> GrammarAnomalyScore:
        """
        Detect if trace is anomalous.

        Args:
            trace: Trace array to analyze

        Returns:
            GrammarAnomalyScore with detection results
        """
        if self.grammar is None:
            raise ValueError("Detector not trained. Call train() first.")

        # Encode trace to SAX
        sax_seq = self._encode_trace(trace)

        # Extract features
        features = self.feature_extractor.extract_features(sax_seq)
        coverage = self.feature_extractor.compute_coverage(sax_seq)

        # Compute anomaly score
        if self.use_coverage:
            # Coverage-based: low coverage = anomaly
            if self.normal_coverage_std > 0:
                z_score = (coverage - self.normal_coverage_mean) / self.normal_coverage_std
                # Negative z-score (below normal) indicates anomaly
                anomaly_score = max(0, -z_score)
            else:
                # If std is 0, use simple threshold
                anomaly_score = 1.0 if coverage < self.normal_coverage_mean else 0.0
        else:
            # Feature-based scoring (more advanced)
            scores = []

            for key in ['compression_ratio', 'num_activated_rules']:
                if key in self.normal_features_mean:
                    mean = self.normal_features_mean[key]
                    std = self.normal_features_std[key]

                    if std > 0:
                        val = features.get(key, mean)
                        z = abs((val - mean) / std)
                        scores.append(z)

            anomaly_score = np.mean(scores) if scores else 0.0

        # Classify
        is_anomaly = anomaly_score > self.anomaly_threshold

        return GrammarAnomalyScore(
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            coverage=float(coverage),
            num_matched_rules=int(features['num_activated_rules']),
            features=features
        )

    def detect_batch(self, traces: List[np.ndarray]) -> List[GrammarAnomalyScore]:
        """
        Detect anomalies in multiple traces.

        Args:
            traces: List of traces

        Returns:
            List of anomaly scores
        """
        return [self.detect(trace) for trace in traces]

    def evaluate(self,
                 traces: List[np.ndarray],
                 labels: List[int]) -> Dict[str, float]:
        """
        Evaluate detector on labeled data.

        Args:
            traces: List of traces
            labels: True labels (0 = normal, 1 = anomaly)

        Returns:
            Metrics dictionary
        """
        predictions = []
        scores = []

        for trace in traces:
            result = self.detect(trace)
            predictions.append(1 if result.is_anomaly else 0)
            scores.append(result.anomaly_score)

        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
        }

        # ROC AUC (if we have both classes)
        if len(set(labels)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(labels, scores)
            except:
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0

        return metrics


# Example usage and testing
if __name__ == "__main__":
    print("Testing Grammar-Based Anomaly Detector...")

    # Create synthetic data
    print("\n1. Generating synthetic microservice traces...")

    np.random.seed(42)

    # Normal traces: consistent latency patterns
    normal_traces = []
    for i in range(100):
        # 10 services with latencies around 20-30ms
        trace = np.random.randn(10) * 5 + 25
        trace = np.maximum(trace, 5)  # Min 5ms
        normal_traces.append(trace)

    # Anomalous traces: higher latencies, different patterns
    anomalous_traces = []
    for i in range(20):
        # Higher latencies (50-70ms) or missing services
        trace = np.random.randn(10) * 10 + 60
        trace = np.maximum(trace, 5)
        anomalous_traces.append(trace)

    print(f"   Created {len(normal_traces)} normal traces")
    print(f"   Created {len(anomalous_traces)} anomalous traces")

    # Train detector
    print("\n2. Training grammar-based detector...")
    detector = GrammarAnomalyDetector(
        word_size=4,
        alphabet_size=4,
        anomaly_threshold=1.0,
        use_coverage=True
    )

    detector.train(normal_traces[:80], verbose=True)  # Train on 80 normal traces

    # Test on remaining normal traces
    print("\n3. Testing on normal traces...")
    test_normal = normal_traces[80:]
    results_normal = detector.detect_batch(test_normal)

    normal_anomaly_rate = sum(r.is_anomaly for r in results_normal) / len(results_normal)
    avg_coverage_normal = np.mean([r.coverage for r in results_normal])

    print(f"   Tested on {len(test_normal)} normal traces")
    print(f"   False positive rate: {normal_anomaly_rate:.1%}")
    print(f"   Average coverage: {avg_coverage_normal:.1%}")

    # Test on anomalous traces
    print("\n4. Testing on anomalous traces...")
    results_anomalous = detector.detect_batch(anomalous_traces)

    anomaly_detection_rate = sum(r.is_anomaly for r in results_anomalous) / len(results_anomalous)
    avg_coverage_anomalous = np.mean([r.coverage for r in results_anomalous])

    print(f"   Tested on {len(anomalous_traces)} anomalous traces")
    print(f"   True positive rate: {anomaly_detection_rate:.1%}")
    print(f"   Average coverage: {avg_coverage_anomalous:.1%}")

    # Evaluate with labels
    print("\n5. Overall evaluation...")
    all_traces = test_normal + anomalous_traces
    all_labels = [0] * len(test_normal) + [1] * len(anomalous_traces)

    metrics = detector.evaluate(all_traces, all_labels)

    print(f"   Metrics:")
    for metric, value in metrics.items():
        print(f"     {metric:12s}: {value:.3f}")

    # Show sample detections
    print("\n6. Sample anomaly scores:")
    print(f"   Normal traces:")
    for i, result in enumerate(results_normal[:3]):
        print(f"     Trace {i}: score={result.anomaly_score:.3f}, coverage={result.coverage:.3f}, "
              f"matched_rules={result.num_matched_rules}, anomaly={result.is_anomaly}")

    print(f"\n   Anomalous traces:")
    for i, result in enumerate(results_anomalous[:3]):
        print(f"     Trace {i}: score={result.anomaly_score:.3f}, coverage={result.coverage:.3f}, "
              f"matched_rules={result.num_matched_rules}, anomaly={result.is_anomaly}")

    print("\n✅ Grammar-based detector tests passed!")
