"""
Trace Parser Module
Parses Jaeger/Zipkin distributed traces into structured format.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class Span:
    """Represents a single span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: int  # microseconds
    duration: int    # microseconds
    tags: Dict
    logs: List[Dict]

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration

    @property
    def duration_ms(self) -> float:
        return self.duration / 1000.0


@dataclass
class Trace:
    """Represents a complete distributed trace."""
    trace_id: str
    spans: List[Span]

    @property
    def root_span(self) -> Optional[Span]:
        """Get the root span (no parent)."""
        for span in self.spans:
            if span.parent_span_id is None:
                return span
        return self.spans[0] if self.spans else None

    @property
    def services(self) -> List[str]:
        """Get unique services in this trace."""
        return list(set(s.service_name for s in self.spans))

    @property
    def duration_ms(self) -> float:
        """Total trace duration."""
        if not self.spans:
            return 0.0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time for s in self.spans)
        return (end - start) / 1000.0

    def get_call_sequence(self) -> List[str]:
        """Get service call sequence ordered by start time."""
        sorted_spans = sorted(self.spans, key=lambda s: s.start_time)
        return [s.service_name for s in sorted_spans]


class TraceParser(ABC):
    """Abstract base class for trace parsers."""

    @abstractmethod
    def parse_file(self, filepath: Path) -> List[Trace]:
        """Parse a single trace file."""
        pass

    @abstractmethod
    def parse_directory(self, dirpath: Path, max_files: Optional[int] = None) -> List[Trace]:
        """Parse all trace files in a directory."""
        pass

    def to_dataframe(self, traces: List[Trace]) -> pd.DataFrame:
        """Convert traces to pandas DataFrame."""
        records = []
        for trace in traces:
            for span in trace.spans:
                records.append({
                    'trace_id': span.trace_id,
                    'span_id': span.span_id,
                    'parent_span_id': span.parent_span_id,
                    'operation_name': span.operation_name,
                    'service_name': span.service_name,
                    'start_time': span.start_time,
                    'duration_us': span.duration,
                    'duration_ms': span.duration_ms,
                    'num_tags': len(span.tags),
                    'num_logs': len(span.logs),
                })
        return pd.DataFrame(records)


class JaegerTraceParser(TraceParser):
    """Parser for Jaeger trace format (JSON)."""

    def __init__(self):
        self.process_map = {}  # Maps processID to service name

    def _parse_span(self, span_data: Dict, trace_id: str) -> Span:
        """Parse a single span from Jaeger format."""
        # Get parent span ID from references
        parent_span_id = None
        references = span_data.get('references', [])
        for ref in references:
            if ref.get('refType') == 'CHILD_OF':
                parent_span_id = ref.get('spanID')
                break

        # Get service name from process
        process_id = span_data.get('processID', '')
        service_name = self.process_map.get(process_id, process_id)

        return Span(
            trace_id=trace_id,
            span_id=span_data.get('spanID', ''),
            parent_span_id=parent_span_id,
            operation_name=span_data.get('operationName', ''),
            service_name=service_name,
            start_time=span_data.get('startTime', 0),
            duration=span_data.get('duration', 0),
            tags={t['key']: t.get('value') for t in span_data.get('tags', [])},
            logs=span_data.get('logs', [])
        )

    def _parse_trace(self, trace_data: Dict) -> Trace:
        """Parse a single trace from Jaeger format."""
        trace_id = trace_data.get('traceID', '')

        # Build process map for this trace
        processes = trace_data.get('processes', {})
        for pid, pdata in processes.items():
            self.process_map[pid] = pdata.get('serviceName', pid)

        # Parse spans
        spans = []
        for span_data in trace_data.get('spans', []):
            spans.append(self._parse_span(span_data, trace_id))

        return Trace(trace_id=trace_id, spans=spans)

    def parse_file(self, filepath: Path) -> List[Trace]:
        """Parse a Jaeger JSON trace file."""
        traces = []
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Handle different Jaeger export formats
            if isinstance(data, dict):
                if 'data' in data:
                    trace_list = data['data']
                else:
                    trace_list = [data]
            elif isinstance(data, list):
                trace_list = data
            else:
                return traces

            for trace_data in trace_list:
                traces.append(self._parse_trace(trace_data))

        except Exception as e:
            print(f"Error parsing {filepath}: {e}")

        return traces

    def parse_directory(self, dirpath: Path, max_files: Optional[int] = None) -> List[Trace]:
        """Parse all JSON trace files in a directory."""
        traces = []
        json_files = list(dirpath.glob('**/*.json'))

        if max_files:
            json_files = json_files[:max_files]

        for filepath in json_files:
            traces.extend(self.parse_file(filepath))

        return traces


class TrainTicketParser(JaegerTraceParser):
    """Specialized parser for Train-Ticket dataset."""

    def __init__(self, data_dir: Path):
        super().__init__()
        self.data_dir = data_dir
        self.traces_dir = data_dir / 'traces'
        self.logs_dir = data_dir / 'logs'
        self.metrics_dir = data_dir / 'metrics'

    def load_all(self, max_traces: Optional[int] = None) -> Tuple[List[Trace], pd.DataFrame]:
        """Load traces and convert to DataFrame."""
        if not self.traces_dir.exists():
            raise FileNotFoundError(f"Traces directory not found: {self.traces_dir}")

        traces = self.parse_directory(self.traces_dir, max_files=max_traces)
        df = self.to_dataframe(traces)

        return traces, df

    def load_labels(self, label_file: Optional[Path] = None) -> pd.DataFrame:
        """Load anomaly labels if available."""
        if label_file is None:
            label_file = self.data_dir / 'labels.csv'

        if label_file.exists():
            return pd.read_csv(label_file)
        return pd.DataFrame()


def extract_call_sequences(traces: List[Trace]) -> List[List[str]]:
    """Extract service call sequences from traces."""
    return [trace.get_call_sequence() for trace in traces]


def extract_latency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract latency-based features per trace."""
    features = df.groupby('trace_id').agg({
        'duration_ms': ['mean', 'std', 'min', 'max', 'sum'],
        'service_name': 'nunique',
        'span_id': 'count'
    }).reset_index()

    features.columns = ['trace_id', 'avg_latency', 'std_latency', 'min_latency',
                        'max_latency', 'total_latency', 'num_services', 'num_spans']

    return features
