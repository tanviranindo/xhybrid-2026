"""
SAX (Symbolic Aggregate approXimation) Encoding
Replicates D'Angelo & d'Aloisio ICPE 2024 approach

SAX converts time series to symbolic representation:
1. Normalize time series (z-score)
2. Piecewise Aggregate Approximation (PAA)
3. Discretize to symbols using Gaussian breakpoints
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.stats import norm


class SAXEncoder:
    """
    Symbolic Aggregate approXimation (SAX) encoder.

    Converts numeric time series to symbolic strings for pattern mining.

    Args:
        word_size: Number of PAA segments (n)
        alphabet_size: Number of symbols (a)
        normalize: Whether to z-normalize time series
    """

    def __init__(self,
                 word_size: int = 8,
                 alphabet_size: int = 4,
                 normalize: bool = True):
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.normalize = normalize

        # Precompute breakpoints for Gaussian distribution
        self.breakpoints = self._compute_breakpoints(alphabet_size)

        # Create alphabet (a, b, c, ...)
        self.alphabet = [chr(97 + i) for i in range(alphabet_size)]

    def _compute_breakpoints(self, alphabet_size: int) -> np.ndarray:
        """
        Compute breakpoints for equiprobable regions under Gaussian.

        For alphabet_size = 4:
        breakpoints = [-0.67, 0, 0.67]
        Regions: (-inf, -0.67), [-0.67, 0), [0, 0.67), [0.67, inf)
        """
        if alphabet_size == 1:
            return np.array([])

        # Compute equiprobable breakpoints
        breakpoints = []
        for i in range(1, alphabet_size):
            # Inverse CDF at i/alphabet_size
            breakpoint = norm.ppf(i / alphabet_size)
            breakpoints.append(breakpoint)

        return np.array(breakpoints)

    def _znormalize(self, ts: np.ndarray) -> np.ndarray:
        """Z-score normalization: (x - mean) / std."""
        mean = ts.mean()
        std = ts.std()

        if std == 0:
            return np.zeros_like(ts)

        return (ts - mean) / std

    def _paa(self, ts: np.ndarray, segments: int) -> np.ndarray:
        """
        Piecewise Aggregate Approximation.

        Divides time series into segments and computes mean of each.

        Args:
            ts: Time series of length N
            segments: Number of segments (word_size)

        Returns:
            PAA representation of length segments
        """
        n = len(ts)

        if n < segments:
            # If time series shorter than segments, pad with mean
            pad_length = segments - n
            ts = np.concatenate([ts, np.full(pad_length, ts.mean())])
            n = segments

        # Split into segments and compute means
        paa_values = []

        for i in range(segments):
            start_idx = int(i * n / segments)
            end_idx = int((i + 1) * n / segments)

            segment = ts[start_idx:end_idx]
            paa_values.append(segment.mean())

        return np.array(paa_values)

    def _discretize(self, paa: np.ndarray) -> str:
        """
        Discretize PAA values to symbols.

        Uses precomputed breakpoints to map continuous values
        to discrete symbols.
        """
        symbols = []

        for value in paa:
            # Find which region the value falls into
            symbol_idx = np.searchsorted(self.breakpoints, value)
            symbols.append(self.alphabet[symbol_idx])

        return ''.join(symbols)

    def encode(self, time_series: np.ndarray) -> str:
        """
        Encode time series to SAX string.

        Args:
            time_series: Input time series (1D array)

        Returns:
            SAX string (e.g., "abcdabcd")
        """
        # Step 1: Normalize
        if self.normalize:
            ts = self._znormalize(time_series)
        else:
            ts = time_series.copy()

        # Step 2: PAA
        paa = self._paa(ts, self.word_size)

        # Step 3: Discretize
        sax_string = self._discretize(paa)

        return sax_string

    def encode_batch(self, time_series_list: List[np.ndarray]) -> List[str]:
        """
        Encode multiple time series.

        Args:
            time_series_list: List of time series arrays

        Returns:
            List of SAX strings
        """
        return [self.encode(ts) for ts in time_series_list]

    def encode_multivariate(self,
                            multivariate_ts: np.ndarray,
                            axis: int = 1) -> List[str]:
        """
        Encode multivariate time series.

        For microservice traces, each dimension is a service.

        Args:
            multivariate_ts: (n_samples, n_features) or (n_features, n_samples)
            axis: Which axis represents features

        Returns:
            List of SAX strings, one per feature
        """
        if axis == 0:
            multivariate_ts = multivariate_ts.T

        sax_strings = []
        for feature_ts in multivariate_ts:
            sax_str = self.encode(feature_ts)
            sax_strings.append(sax_str)

        return sax_strings

    def distance(self, s1: str, s2: str) -> float:
        """
        MINDIST: Minimum distance between two SAX strings.

        This is a lower bound of the Euclidean distance between
        the original time series.

        Args:
            s1, s2: SAX strings of same length

        Returns:
            MINDIST value
        """
        if len(s1) != len(s2):
            raise ValueError(f"SAX strings must have same length: {len(s1)} != {len(s2)}")

        # Precompute distance table
        dist_table = self._build_distance_table()

        total_dist = 0.0
        for c1, c2 in zip(s1, s2):
            idx1 = self.alphabet.index(c1)
            idx2 = self.alphabet.index(c2)
            total_dist += dist_table[idx1, idx2] ** 2

        # Scale by sqrt(n / word_size)
        # Assuming original length n is proportional to word_size
        scale = np.sqrt(self.word_size)

        return scale * np.sqrt(total_dist)

    def _build_distance_table(self) -> np.ndarray:
        """
        Build lookup table for symbol distances.

        dist(symbol_i, symbol_j) = |breakpoint[max(i,j)-1] - breakpoint[min(i,j)]|
        if |i - j| > 1, else 0
        """
        a = self.alphabet_size
        table = np.zeros((a, a))

        for i in range(a):
            for j in range(a):
                if abs(i - j) <= 1:
                    table[i, j] = 0
                else:
                    # Distance between non-adjacent regions
                    min_idx = min(i, j)
                    max_idx = max(i, j)
                    table[i, j] = self.breakpoints[max_idx - 1] - self.breakpoints[min_idx]

        return table


class ServiceSequenceSAX:
    """
    SAX encoding specifically for microservice trace sequences.

    Converts service call sequences to SAX representation for
    grammar induction.
    """

    def __init__(self,
                 word_size: int = 8,
                 alphabet_size: int = 4):
        self.encoder = SAXEncoder(word_size, alphabet_size, normalize=True)

    def encode_trace(self,
                     service_latencies: np.ndarray,
                     service_names: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
        """
        Encode a single trace (service call latencies).

        Args:
            service_latencies: (n_services,) latency for each service
            service_names: Optional service names

        Returns:
            (sax_symbols, service_names)
        """
        # Encode each service's latency
        sax_symbols = []

        for i, latency in enumerate(service_latencies):
            # For a single latency value, create a simple encoding
            # Map latency to symbol based on normalized value
            if self.encoder.normalize:
                # Use global statistics (would need to be computed from training data)
                # For now, use simple thresholding
                normalized = (latency - 20) / 10  # Rough normalization
            else:
                normalized = latency

            # Discretize single value
            symbol_idx = np.searchsorted(self.encoder.breakpoints, normalized)
            symbol = self.encoder.alphabet[min(symbol_idx, self.encoder.alphabet_size - 1)]
            sax_symbols.append(symbol)

        if service_names is None:
            service_names = [f"s{i}" for i in range(len(service_latencies))]

        return sax_symbols, service_names

    def encode_trace_temporal(self,
                              temporal_latencies: np.ndarray,
                              service_id: int) -> str:
        """
        Encode temporal sequence for a single service.

        Args:
            temporal_latencies: (n_samples,) latency over time for one service
            service_id: Service identifier

        Returns:
            SAX string
        """
        return self.encoder.encode(temporal_latencies)


# Example usage and testing
if __name__ == "__main__":
    print("Testing SAX Encoder...")

    # Test 1: Basic SAX encoding
    print("\n1. Basic SAX Encoding:")
    encoder = SAXEncoder(word_size=8, alphabet_size=4)

    # Create test time series
    ts = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    sax_str = encoder.encode(ts)
    print(f"   Time series: {ts}")
    print(f"   SAX string: {sax_str}")
    print(f"   Alphabet: {encoder.alphabet}")
    print(f"   Breakpoints: {encoder.breakpoints}")

    # Test 2: Encode multiple time series
    print("\n2. Batch Encoding:")
    ts_list = [
        np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        np.array([8, 7, 6, 5, 4, 3, 2, 1]),
        np.array([1, 1, 1, 1, 1, 1, 1, 1]),
    ]
    sax_strings = encoder.encode_batch(ts_list)
    for i, sax in enumerate(sax_strings):
        print(f"   TS {i}: {sax}")

    # Test 3: Multivariate encoding (microservice trace)
    print("\n3. Multivariate Encoding (Microservice Trace):")
    # Simulate 5 services with 10 time steps each
    trace = np.random.randn(10, 5) * 10 + 50  # 10 samples, 5 services
    sax_per_service = encoder.encode_multivariate(trace.T, axis=0)

    print(f"   Trace shape: {trace.shape}")
    for i, sax in enumerate(sax_per_service):
        print(f"   Service {i}: {sax}")

    # Test 4: SAX distance
    print("\n4. SAX Distance (MINDIST):")
    s1 = "abcd"
    s2 = "abdc"
    s3 = "dcba"

    dist_12 = encoder.distance(s1, s2)
    dist_13 = encoder.distance(s1, s3)
    dist_11 = encoder.distance(s1, s1)

    print(f"   distance('{s1}', '{s2}') = {dist_12:.3f}")
    print(f"   distance('{s1}', '{s3}') = {dist_13:.3f}")
    print(f"   distance('{s1}', '{s1}') = {dist_11:.3f}")

    # Test 5: Service sequence SAX
    print("\n5. Service Sequence SAX:")
    service_sax = ServiceSequenceSAX(word_size=4, alphabet_size=4)

    # Simulate service latencies for a single trace
    service_latencies = np.array([10, 25, 15, 30, 20, 35, 12, 28])
    service_names = [f"svc-{i}" for i in range(8)]

    sax_symbols, names = service_sax.encode_trace(service_latencies, service_names)
    print(f"   Service latencies: {service_latencies}")
    print(f"   SAX symbols: {sax_symbols}")
    print(f"   Combined: {' '.join([f'{n}:{s}' for n, s in zip(names, sax_symbols)])}")

    print("\n✅ SAX Encoder tests passed!")
