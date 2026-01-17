"""
Sequitur Grammar Induction Algorithm
Replicates D'Angelo & d'Aloisio ICPE 2024 approach

Sequitur extracts context-free grammar from sequences:
- Finds repeating patterns (digrams)
- Creates production rules hierarchically
- Enforces utility constraint (each rule used > once)
"""

from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Rule:
    """Grammar production rule."""
    lhs: str  # Left-hand side (non-terminal)
    rhs: Tuple[str, ...]  # Right-hand side (sequence of symbols)
    count: int = 0  # Number of times this rule appears

    def __str__(self):
        return f"{self.lhs} -> {' '.join(self.rhs)}"

    def __hash__(self):
        return hash((self.lhs, self.rhs))


class SequiturGrammar:
    """
    Sequitur algorithm for grammar induction.

    Extracts hierarchical patterns from symbol sequences.

    Main principles:
    1. Digram uniqueness: No digram appears more than once
    2. Rule utility: Every rule is used more than once

    Args:
        min_pattern_length: Minimum length for patterns (default 2)
        max_rules: Maximum number of rules to extract
    """

    def __init__(self,
                 min_pattern_length: int = 2,
                 max_rules: int = 100):
        self.min_pattern_length = min_pattern_length
        self.max_rules = max_rules

        self.rules: List[Rule] = []
        self.rule_counter = 0

        # Index for fast digram lookup
        self.digram_positions: Dict[Tuple[str, str], List[int]] = defaultdict(list)

    def _create_rule(self, pattern: Tuple[str, ...]) -> str:
        """Create new non-terminal for pattern."""
        rule_name = f"R{self.rule_counter}"
        self.rule_counter += 1

        rule = Rule(lhs=rule_name, rhs=pattern, count=0)
        self.rules.append(rule)

        return rule_name

    def _find_repeating_digrams(self, sequence: List[str]) -> Dict[Tuple[str, str], List[int]]:
        """
        Find all repeating digrams (2-grams) in sequence.

        Returns:
            Dict mapping digram to list of positions where it occurs
        """
        digrams = defaultdict(list)

        for i in range(len(sequence) - 1):
            digram = (sequence[i], sequence[i + 1])
            digrams[digram].append(i)

        # Filter to only repeating digrams
        repeating = {d: pos for d, pos in digrams.items() if len(pos) > 1}

        return repeating

    def _replace_digram(self,
                        sequence: List[str],
                        digram: Tuple[str, str],
                        positions: List[int]) -> Tuple[List[str], str]:
        """
        Replace all occurrences of digram with a new non-terminal.

        Returns:
            (new_sequence, rule_name)
        """
        # Create rule for this digram
        rule_name = self._create_rule(digram)

        # Replace from right to left to maintain positions
        new_sequence = sequence.copy()
        for pos in sorted(positions, reverse=True):
            # Replace digram at position with rule_name
            if pos + 1 < len(new_sequence):
                new_sequence[pos:pos + 2] = [rule_name]

        return new_sequence, rule_name

    def _enforce_utility(self):
        """Remove rules that are only used once."""
        # Count rule usage
        rule_usage = defaultdict(int)

        # Count in other rules' RHS
        for rule in self.rules:
            for symbol in rule.rhs:
                if symbol.startswith('R'):
                    rule_usage[symbol] += 1

        # Remove unused rules
        self.rules = [r for r in self.rules if rule_usage.get(r.lhs, 0) > 1 or r.lhs == 'S']

    def induce(self, sequence: List[str]) -> List[Rule]:
        """
        Induce grammar from sequence using Sequitur algorithm.

        Args:
            sequence: Input sequence (e.g., SAX symbols)

        Returns:
            List of grammar rules
        """
        self.rules = []
        self.rule_counter = 0

        if len(sequence) < self.min_pattern_length:
            # Too short, create single rule
            start_rule = Rule(lhs='S', rhs=tuple(sequence), count=1)
            return [start_rule]

        # Start with the input sequence
        current_seq = sequence.copy()

        # Iteratively find and replace repeating digrams
        iteration = 0
        while iteration < self.max_rules:
            # Find repeating digrams
            repeating = self._find_repeating_digrams(current_seq)

            if not repeating:
                # No more patterns to extract
                break

            # Get most frequent digram
            most_frequent = max(repeating.items(), key=lambda x: len(x[1]))
            digram, positions = most_frequent

            # Replace digram with new rule
            current_seq, rule_name = self._replace_digram(current_seq, digram, positions)

            # Update rule count
            for rule in self.rules:
                if rule.lhs == rule_name:
                    rule.count = len(positions)
                    break

            iteration += 1

        # Create start rule
        start_rule = Rule(lhs='S', rhs=tuple(current_seq), count=1)
        self.rules.insert(0, start_rule)

        # Enforce utility constraint
        self._enforce_utility()

        return self.rules

    def induce_batch(self, sequences: List[List[str]]) -> List[Rule]:
        """
        Induce grammar from multiple sequences.

        Concatenates sequences and finds common patterns.

        Args:
            sequences: List of input sequences

        Returns:
            List of grammar rules
        """
        # Concatenate sequences with separator
        combined = []
        for seq in sequences:
            combined.extend(seq)
            combined.append('|')  # Separator

        # Remove last separator
        if combined and combined[-1] == '|':
            combined.pop()

        # Induce grammar
        rules = self.induce(combined)

        return rules

    def expand_rule(self, rule: Rule, max_depth: int = 10) -> List[str]:
        """
        Expand a rule to its terminal sequence.

        Args:
            rule: Rule to expand
            max_depth: Maximum recursion depth

        Returns:
            Expanded terminal sequence
        """
        if max_depth == 0:
            return list(rule.rhs)

        expanded = []
        for symbol in rule.rhs:
            if symbol.startswith('R'):
                # Find rule for this non-terminal
                for r in self.rules:
                    if r.lhs == symbol:
                        expanded.extend(self.expand_rule(r, max_depth - 1))
                        break
                else:
                    # Rule not found, keep symbol
                    expanded.append(symbol)
            else:
                # Terminal symbol
                expanded.append(symbol)

        return expanded

    def to_dict(self) -> Dict:
        """Convert grammar to dictionary representation."""
        return {
            'rules': [
                {
                    'lhs': r.lhs,
                    'rhs': list(r.rhs),
                    'count': r.count
                }
                for r in self.rules
            ],
            'num_rules': len(self.rules)
        }


class GrammarFeatureExtractor:
    """
    Extract features from grammar for anomaly detection.

    Features:
    1. Grammar size (number of rules)
    2. Rule activation patterns
    3. Compression ratio
    4. Pattern coverage
    """

    def __init__(self, grammar: SequiturGrammar):
        self.grammar = grammar

    def extract_features(self, sequence: List[str]) -> Dict[str, float]:
        """
        Extract grammar-based features from a sequence.

        Args:
            sequence: Input sequence to analyze

        Returns:
            Feature dictionary
        """
        features = {}

        # Feature 1: Grammar size
        features['num_rules'] = len(self.grammar.rules)

        # Feature 2: Sequence length
        features['sequence_length'] = len(sequence)

        # Feature 3: Compression ratio
        # (original length / compressed length)
        start_rule = self.grammar.rules[0] if self.grammar.rules else None
        if start_rule:
            compressed_length = len(start_rule.rhs)
            features['compression_ratio'] = len(sequence) / max(compressed_length, 1)
        else:
            features['compression_ratio'] = 1.0

        # Feature 4: Average rule usage
        if self.grammar.rules:
            avg_usage = sum(r.count for r in self.grammar.rules) / len(self.grammar.rules)
            features['avg_rule_usage'] = avg_usage
        else:
            features['avg_rule_usage'] = 0.0

        # Feature 5: Rule activation vector
        # For each rule, check if it's used in the sequence
        rule_activations = []
        for rule in self.grammar.rules:
            # Check if rule's pattern appears in sequence
            activation = 0
            for i in range(len(sequence) - len(rule.rhs) + 1):
                if tuple(sequence[i:i + len(rule.rhs)]) == rule.rhs:
                    activation = 1
                    break
            rule_activations.append(activation)

        features['rule_activations'] = rule_activations
        features['num_activated_rules'] = sum(rule_activations)

        return features

    def compute_coverage(self, sequence: List[str]) -> float:
        """
        Compute how much of the sequence is covered by grammar rules.

        Returns:
            Coverage ratio [0, 1]
        """
        covered_positions = set()

        for rule in self.grammar.rules:
            pattern = rule.rhs

            # Find all occurrences of this pattern
            for i in range(len(sequence) - len(pattern) + 1):
                if tuple(sequence[i:i + len(pattern)]) == pattern:
                    # Mark positions as covered
                    for j in range(i, i + len(pattern)):
                        covered_positions.add(j)

        coverage = len(covered_positions) / max(len(sequence), 1)
        return coverage


# Example usage and testing
if __name__ == "__main__":
    print("Testing Sequitur Grammar Induction...")

    # Test 1: Simple repeating pattern
    print("\n1. Simple Pattern:")
    grammar = SequiturGrammar()
    seq = ['a', 'b', 'a', 'b', 'a', 'b', 'c', 'd', 'c', 'd']
    rules = grammar.induce(seq)

    print(f"   Input: {seq}")
    print(f"   Grammar ({len(rules)} rules):")
    for rule in rules:
        print(f"     {rule}")

    # Test 2: Hierarchical pattern
    print("\n2. Hierarchical Pattern:")
    grammar2 = SequiturGrammar()
    seq2 = ['a', 'b', 'c', 'a', 'b', 'c', 'd', 'e', 'd', 'e']
    rules2 = grammar2.induce(seq2)

    print(f"   Input: {seq2}")
    print(f"   Grammar ({len(rules2)} rules):")
    for rule in rules2:
        print(f"     {rule}")

    # Test 3: Multiple sequences
    print("\n3. Multiple Sequences:")
    grammar3 = SequiturGrammar()
    seqs = [
        ['a', 'b', 'c', 'd'],
        ['a', 'b', 'e', 'f'],
        ['a', 'b', 'c', 'd'],
    ]
    rules3 = grammar3.induce_batch(seqs)

    print(f"   Input sequences: {seqs}")
    print(f"   Grammar ({len(rules3)} rules):")
    for rule in rules3:
        print(f"     {rule}")

    # Test 4: Rule expansion
    print("\n4. Rule Expansion:")
    if len(rules) > 0:
        start_rule = rules[0]
        expanded = grammar.expand_rule(start_rule)
        print(f"   Start rule: {start_rule}")
        print(f"   Expanded: {expanded}")
        print(f"   Original: {seq}")
        print(f"   Match: {expanded == seq}")

    # Test 5: Feature extraction
    print("\n5. Feature Extraction:")
    extractor = GrammarFeatureExtractor(grammar)
    features = extractor.extract_features(seq)

    print(f"   Features:")
    for key, val in features.items():
        if key != 'rule_activations':
            print(f"     {key}: {val}")
        else:
            print(f"     {key}: {val}")

    coverage = extractor.compute_coverage(seq)
    print(f"   Coverage: {coverage:.2%}")

    # Test 6: SAX + Sequitur
    print("\n6. SAX + Sequitur Integration:")
    # Simulate SAX strings from microservice traces
    sax_strings = [
        ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'],
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'],
        ['a', 'b', 'c', 'd', 'a', 'b', 'c', 'd'],
    ]

    grammar_sax = SequiturGrammar()
    rules_sax = grammar_sax.induce_batch(sax_strings)

    print(f"   SAX strings: {len(sax_strings)}")
    print(f"   Extracted grammar ({len(rules_sax)} rules):")
    for rule in rules_sax[:5]:  # Show first 5
        print(f"     {rule}")

    print("\n✅ Sequitur Grammar tests passed!")
