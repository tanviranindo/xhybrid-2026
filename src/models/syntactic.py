"""
Syntactic Module for Pattern Recognition
Implements SAX encoding and grammar-based pattern inference for microservice traces.
Based on D'Angelo & d'Aloisio (2024) with neural integration extensions.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
import re


@dataclass
class GrammarRule:
    """Represents a grammar production rule."""
    lhs: str  # Left-hand side (non-terminal)
    rhs: Tuple[str, ...]  # Right-hand side (sequence of symbols)
    frequency: int = 0
    support: float = 0.0

    def __str__(self):
        return f"{self.lhs} -> {' '.join(self.rhs)}"

    def __hash__(self):
        return hash((self.lhs, self.rhs))


class SAXEncoder:
    """
    Symbolic Aggregate approXimation (SAX) encoder for time series.
    Converts latency sequences into symbolic representations.
    """

    def __init__(self,
                 alphabet_size: int = 8,
                 word_size: int = 10,
                 normalize: bool = True):
        """
        Args:
            alphabet_size: Number of symbols in alphabet (2-26)
            word_size: Number of PAA segments
            normalize: Whether to z-normalize sequences
        """
        self.alphabet_size = min(max(alphabet_size, 2), 26)
        self.word_size = word_size
        self.normalize = normalize

        # Generate breakpoints for Gaussian distribution
        self.breakpoints = self._generate_breakpoints()
        self.alphabet = [chr(ord('a') + i) for i in range(self.alphabet_size)]

    def _generate_breakpoints(self) -> np.ndarray:
        """Generate equiprobable breakpoints for Gaussian distribution."""
        from scipy.stats import norm
        breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size + 1)[1:-1])
        return breakpoints

    def _paa(self, series: np.ndarray) -> np.ndarray:
        """Piecewise Aggregate Approximation."""
        n = len(series)
        if n == self.word_size:
            return series

        # Compute PAA representation
        paa_values = np.zeros(self.word_size)
        for i in range(self.word_size):
            start = int(i * n / self.word_size)
            end = int((i + 1) * n / self.word_size)
            if end > start:
                paa_values[i] = np.mean(series[start:end])

        return paa_values

    def _discretize(self, paa_values: np.ndarray) -> str:
        """Convert PAA values to SAX string."""
        sax_word = []
        for val in paa_values:
            idx = np.searchsorted(self.breakpoints, val)
            sax_word.append(self.alphabet[idx])
        return ''.join(sax_word)

    def encode(self, series: np.ndarray) -> str:
        """
        Encode a time series into SAX representation.

        Args:
            series: 1D numpy array of numerical values

        Returns:
            SAX string representation
        """
        if len(series) == 0:
            return 'a' * self.word_size

        # Filter out zeros (padding)
        non_zero = series[series > 0]
        if len(non_zero) == 0:
            return 'a' * self.word_size

        series = non_zero

        # Z-normalize
        if self.normalize:
            mean = np.mean(series)
            std = np.std(series)
            if std > 0:
                series = (series - mean) / std
            else:
                series = series - mean

        # PAA
        paa = self._paa(series)

        # Discretize
        return self._discretize(paa)

    def encode_batch(self, sequences: np.ndarray) -> List[str]:
        """Encode multiple sequences."""
        return [self.encode(seq) for seq in sequences]

    def distance(self, sax1: str, sax2: str) -> float:
        """
        Compute MINDIST distance between two SAX words.
        Lower bound of Euclidean distance.
        """
        if len(sax1) != len(sax2):
            raise ValueError("SAX words must have same length")

        # Build distance lookup table
        dist_table = self._build_dist_table()

        n = len(sax1)
        dist_sq = 0.0
        for i in range(n):
            c1, c2 = sax1[i], sax2[i]
            dist_sq += dist_table[(c1, c2)] ** 2

        return np.sqrt(dist_sq)

    def _build_dist_table(self) -> Dict[Tuple[str, str], float]:
        """Build distance lookup table for symbol pairs."""
        table = {}
        beta = self.breakpoints

        for i, c1 in enumerate(self.alphabet):
            for j, c2 in enumerate(self.alphabet):
                if abs(i - j) <= 1:
                    table[(c1, c2)] = 0.0
                else:
                    table[(c1, c2)] = beta[max(i, j) - 1] - beta[min(i, j)]

        return table


class GrammarInference:
    """
    Grammar inference for service call sequences.
    Extracts recurring patterns as grammar rules.
    """

    def __init__(self,
                 min_support: float = 0.05,
                 max_pattern_len: int = 5,
                 min_pattern_len: int = 2):
        """
        Args:
            min_support: Minimum support threshold for pattern extraction
            max_pattern_len: Maximum pattern length
            min_pattern_len: Minimum pattern length
        """
        self.min_support = min_support
        self.max_pattern_len = max_pattern_len
        self.min_pattern_len = min_pattern_len

        self.grammar_rules: List[GrammarRule] = []
        self.pattern_vocab: Dict[Tuple[str, ...], int] = {}
        self.service_vocab: Dict[str, int] = {}

    def fit(self, sequences: List[List[str]]) -> 'GrammarInference':
        """
        Learn grammar rules from service call sequences.

        Args:
            sequences: List of service call sequences
        """
        # Build service vocabulary
        all_services = set()
        for seq in sequences:
            all_services.update(seq)
        self.service_vocab = {s: i for i, s in enumerate(sorted(all_services))}

        # Extract frequent patterns
        pattern_counts = self._extract_patterns(sequences)

        # Convert to grammar rules
        n_sequences = len(sequences)
        rule_id = 0

        for pattern, count in pattern_counts.items():
            support = count / n_sequences
            if support >= self.min_support:
                rule = GrammarRule(
                    lhs=f"P{rule_id}",
                    rhs=pattern,
                    frequency=count,
                    support=support
                )
                self.grammar_rules.append(rule)
                self.pattern_vocab[pattern] = rule_id
                rule_id += 1

        # Sort rules by support
        self.grammar_rules.sort(key=lambda r: r.support, reverse=True)

        return self

    def _extract_patterns(self, sequences: List[List[str]]) -> Dict[Tuple[str, ...], int]:
        """Extract n-gram patterns from sequences."""
        pattern_counts = Counter()

        for seq in sequences:
            # Extract n-grams of different lengths
            for n in range(self.min_pattern_len, self.max_pattern_len + 1):
                for i in range(len(seq) - n + 1):
                    pattern = tuple(seq[i:i + n])
                    pattern_counts[pattern] += 1

        return pattern_counts

    def encode_sequence(self, sequence: List[str]) -> np.ndarray:
        """
        Encode a service call sequence using learned grammar rules.

        Args:
            sequence: Service call sequence

        Returns:
            Binary vector indicating which patterns are present
        """
        if not self.grammar_rules:
            return np.zeros(1)

        encoding = np.zeros(len(self.grammar_rules))

        for i, rule in enumerate(self.grammar_rules):
            if self._pattern_in_sequence(rule.rhs, sequence):
                encoding[i] = 1.0

        return encoding

    def _pattern_in_sequence(self, pattern: Tuple[str, ...], sequence: List[str]) -> bool:
        """Check if pattern exists in sequence."""
        pattern_len = len(pattern)
        for i in range(len(sequence) - pattern_len + 1):
            if tuple(sequence[i:i + pattern_len]) == pattern:
                return True
        return False

    def get_rule_activations(self, sequence: List[str]) -> List[Tuple[GrammarRule, bool]]:
        """Get activation status of each rule for explainability."""
        activations = []
        for rule in self.grammar_rules:
            is_active = self._pattern_in_sequence(rule.rhs, sequence)
            activations.append((rule, is_active))
        return activations

    def num_rules(self) -> int:
        return len(self.grammar_rules)


class SyntacticEncoder(nn.Module):
    """
    Neural network wrapper for syntactic features.
    Combines SAX encodings and grammar patterns into learnable embeddings.
    """

    def __init__(self,
                 sax_dim: int = 64,
                 grammar_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 128,
                 num_grammar_rules: int = 100,
                 sax_alphabet_size: int = 8,
                 sax_word_size: int = 10,
                 dropout: float = 0.1):
        """
        Args:
            sax_dim: Embedding dimension for SAX patterns
            grammar_dim: Dimension for grammar rule embeddings
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_grammar_rules: Maximum number of grammar rules
            sax_alphabet_size: SAX alphabet size
            sax_word_size: SAX word length
            dropout: Dropout rate
        """
        super().__init__()

        self.sax_encoder = SAXEncoder(
            alphabet_size=sax_alphabet_size,
            word_size=sax_word_size
        )

        # Embedding for SAX characters
        self.sax_embedding = nn.Embedding(
            num_embeddings=sax_alphabet_size,
            embedding_dim=sax_dim
        )

        # SAX sequence encoder
        self.sax_lstm = nn.LSTM(
            input_size=sax_dim,
            hidden_size=sax_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Grammar pattern encoder
        self.grammar_fc = nn.Sequential(
            nn.Linear(num_grammar_rules, grammar_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(grammar_dim, grammar_dim)
        )

        # Combined encoder
        combined_dim = sax_dim * 2 + grammar_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_dim = output_dim
        self.num_grammar_rules = num_grammar_rules

    def encode_sax(self, latency_sequences: torch.Tensor) -> torch.Tensor:
        """
        Encode latency sequences using SAX.

        Args:
            latency_sequences: (batch_size, seq_len) tensor of latencies

        Returns:
            (batch_size, sax_dim * 2) SAX embeddings
        """
        batch_size = latency_sequences.size(0)
        device = latency_sequences.device

        # Convert to SAX strings
        latencies_np = latency_sequences.cpu().numpy()
        sax_strings = self.sax_encoder.encode_batch(latencies_np)

        # Convert SAX strings to indices
        sax_indices = []
        for sax_str in sax_strings:
            indices = [ord(c) - ord('a') for c in sax_str]
            sax_indices.append(indices)

        sax_tensor = torch.tensor(sax_indices, dtype=torch.long, device=device)

        # Embed and encode
        sax_embedded = self.sax_embedding(sax_tensor)  # (batch, sax_word_size, sax_dim)
        _, (h_n, _) = self.sax_lstm(sax_embedded)

        # Concatenate forward and backward hidden states
        sax_encoding = torch.cat([h_n[0], h_n[1]], dim=-1)  # (batch, sax_dim * 2)

        return sax_encoding

    def forward(self,
                latency_sequences: torch.Tensor,
                grammar_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining SAX and grammar features.

        Args:
            latency_sequences: (batch_size, seq_len) latency values
            grammar_features: (batch_size, num_grammar_rules) binary pattern indicators

        Returns:
            (batch_size, output_dim) syntactic embeddings
        """
        # Encode SAX
        sax_encoding = self.encode_sax(latency_sequences)

        # Encode grammar patterns
        # Pad grammar features if needed
        if grammar_features.size(1) < self.num_grammar_rules:
            padding = torch.zeros(
                grammar_features.size(0),
                self.num_grammar_rules - grammar_features.size(1),
                device=grammar_features.device
            )
            grammar_features = torch.cat([grammar_features, padding], dim=1)
        elif grammar_features.size(1) > self.num_grammar_rules:
            grammar_features = grammar_features[:, :self.num_grammar_rules]

        grammar_encoding = self.grammar_fc(grammar_features)

        # Fuse encodings
        combined = torch.cat([sax_encoding, grammar_encoding], dim=-1)
        output = self.fusion(combined)

        return output


class SyntacticFeatureExtractor:
    """
    End-to-end syntactic feature extraction pipeline.
    Combines SAX encoding and grammar inference.
    """

    def __init__(self,
                 alphabet_size: int = 8,
                 word_size: int = 10,
                 min_support: float = 0.05,
                 max_pattern_len: int = 5):

        self.sax_encoder = SAXEncoder(
            alphabet_size=alphabet_size,
            word_size=word_size
        )

        self.grammar = GrammarInference(
            min_support=min_support,
            max_pattern_len=max_pattern_len
        )

        self._is_fitted = False

    def fit(self, service_sequences: List[List[str]]) -> 'SyntacticFeatureExtractor':
        """Fit grammar rules on training sequences."""
        self.grammar.fit(service_sequences)
        self._is_fitted = True
        return self

    def transform(self,
                  latency_sequences: np.ndarray,
                  service_sequences: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform sequences into syntactic features.

        Args:
            latency_sequences: (n_samples, seq_len) latency values
            service_sequences: List of service call sequences

        Returns:
            (sax_features, grammar_features) tuple of numpy arrays
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        # SAX encoding
        sax_strings = self.sax_encoder.encode_batch(latency_sequences)
        sax_features = np.array([
            [ord(c) - ord('a') for c in s] for s in sax_strings
        ], dtype=np.int64)

        # Grammar encoding
        grammar_features = np.array([
            self.grammar.encode_sequence(seq) for seq in service_sequences
        ], dtype=np.float32)

        return sax_features, grammar_features

    def fit_transform(self,
                      latency_sequences: np.ndarray,
                      service_sequences: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(service_sequences)
        return self.transform(latency_sequences, service_sequences)

    def get_pattern_explanations(self,
                                  service_sequence: List[str],
                                  top_k: int = 5) -> List[Dict]:
        """
        Get explainable pattern activations for a sequence.

        Args:
            service_sequence: Service call sequence
            top_k: Number of top patterns to return

        Returns:
            List of pattern explanation dictionaries
        """
        activations = self.grammar.get_rule_activations(service_sequence)

        explanations = []
        for rule, is_active in activations:
            if is_active:
                explanations.append({
                    'pattern': ' -> '.join(rule.rhs),
                    'rule_id': rule.lhs,
                    'support': rule.support,
                    'frequency': rule.frequency,
                    'is_active': True
                })

        # Sort by support and return top-k
        explanations.sort(key=lambda x: x['support'], reverse=True)
        return explanations[:top_k]

    def get_anomaly_pattern_deviations(self,
                                        normal_sequence: List[str],
                                        anomaly_sequence: List[str]) -> Dict:
        """
        Compare pattern activations between normal and anomaly sequences.
        Useful for explainability.
        """
        normal_acts = set()
        anomaly_acts = set()

        for rule, is_active in self.grammar.get_rule_activations(normal_sequence):
            if is_active:
                normal_acts.add(rule.lhs)

        for rule, is_active in self.grammar.get_rule_activations(anomaly_sequence):
            if is_active:
                anomaly_acts.add(rule.lhs)

        return {
            'missing_patterns': list(normal_acts - anomaly_acts),
            'unexpected_patterns': list(anomaly_acts - normal_acts),
            'common_patterns': list(normal_acts & anomaly_acts),
            'deviation_score': len(normal_acts.symmetric_difference(anomaly_acts)) /
                              max(len(normal_acts | anomaly_acts), 1)
        }
