"""
Grammar Feature Extractor for Hybrid Model
Combines SAX encoding + Sequitur grammar for anomaly detection
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from .sax_encoding import SAXEncoder
from .sequitur_grammar import SequiturGrammar, GrammarFeatureExtractor


class HybridGrammarFeatures:
    """
    Extract grammar-based features for hybrid anomaly detection.
    
    Pipeline:
    1. Convert service latencies to SAX symbols
    2. Build grammar from normal traces
    3. Extract features: compression ratio, rule matches, coverage
    """
    
    def __init__(self,
                 n_services: int,
                 word_size: int = 8,
                 alphabet_size: int = 4,
                 feature_dim: int = 100):
        self.n_services = n_services
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.feature_dim = feature_dim
        
        # SAX encoder
        self.sax_encoder = SAXEncoder(word_size, alphabet_size, normalize=True)
        
        # Grammar (to be trained)
        self.grammar = None
        self.grammar_extractor = None
        
        # Statistics for normalization
        self.latency_mean = None
        self.latency_std = None
        
    def fit(self, normal_traces: np.ndarray):
        """
        Build grammar from normal traces.
        
        Args:
            normal_traces: (n_samples, n_services) normal latency traces
        """
        print(f"Building grammar from {len(normal_traces)} normal traces...")
        
        # Compute statistics for normalization
        self.latency_mean = normal_traces.mean(axis=0)
        self.latency_std = normal_traces.std(axis=0) + 1e-8
        
        # Convert traces to SAX sequences
        sax_sequences = []
        for trace in normal_traces[:1000]:  # Use subset for speed
            sax_seq = self._trace_to_sax(trace)
            sax_sequences.append(sax_seq)
        
        # Build grammar
        self.grammar = SequiturGrammar(min_pattern_length=2, max_rules=50)
        self.grammar.induce_batch(sax_sequences)
        
        self.grammar_extractor = GrammarFeatureExtractor(self.grammar)
        
        print(f"  Grammar built: {len(self.grammar.rules)} rules")
        
    def _trace_to_sax(self, trace: np.ndarray) -> List[str]:
        """Convert single trace to SAX sequence."""
        # Normalize
        if self.latency_mean is not None:
            normalized = (trace - self.latency_mean) / self.latency_std
        else:
            normalized = trace
        
        # Convert each service latency to symbol
        sax_seq = []
        for val in normalized:
            symbol_idx = np.searchsorted(self.sax_encoder.breakpoints, val)
            symbol_idx = min(symbol_idx, self.alphabet_size - 1)
            symbol = self.sax_encoder.alphabet[symbol_idx]
            sax_seq.append(symbol)
        
        return sax_seq
    
    def extract_features(self, trace: np.ndarray) -> np.ndarray:
        """
        Extract grammar features from a single trace.
        
        Args:
            trace: (n_services,) service latencies
            
        Returns:
            (feature_dim,) feature vector
        """
        if self.grammar is None:
            # Not fitted, return zeros
            return np.zeros(self.feature_dim)
        
        # Convert to SAX
        sax_seq = self._trace_to_sax(trace)
        
        # Extract grammar features
        features_dict = self.grammar_extractor.extract_features(sax_seq)
        coverage = self.grammar_extractor.compute_coverage(sax_seq)
        
        # Build feature vector
        feature_vec = np.zeros(self.feature_dim)
        
        # Basic features
        feature_vec[0] = features_dict['num_rules'] / 50.0  # Normalized
        feature_vec[1] = features_dict['compression_ratio']
        feature_vec[2] = features_dict['avg_rule_usage']
        feature_vec[3] = features_dict['num_activated_rules'] / max(features_dict['num_rules'], 1)
        feature_vec[4] = coverage
        
        # Rule activation vector (pad/truncate to fit)
        activations = features_dict['rule_activations']
        n_act = min(len(activations), self.feature_dim - 5)
        feature_vec[5:5+n_act] = activations[:n_act]
        
        return feature_vec
    
    def extract_batch(self, traces: np.ndarray) -> torch.Tensor:
        """
        Extract features for batch of traces.
        
        Args:
            traces: (batch_size, n_services) traces
            
        Returns:
            (batch_size, feature_dim) feature tensor
        """
        features = []
        for trace in traces:
            feat = self.extract_features(trace)
            features.append(feat)
        
        return torch.from_numpy(np.array(features)).float()
