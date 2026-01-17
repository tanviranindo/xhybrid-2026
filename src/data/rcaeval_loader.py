"""RCAEval Dataset Loader - Adapts RCAEval format to existing pipeline."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class RCAEvalLoader:
    """Load RCAEval datasets (RE1/RE2/RE3) for Train-Ticket system."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
    
    def load_traces(self, dataset: str, scenario: str, run: int = 1) -> Tuple[pd.DataFrame, int]:
        """Load traces and inject_time for a scenario.
        
        Args:
            dataset: 'RE1-TT', 'RE2-TT', or 'RE3-TT'
            scenario: e.g., 'ts-auth-service_cpu'
            run: Run number (1-5)
        
        Returns:
            (traces_df, inject_time) - inject_time is fault injection timestamp
        """
        path = self.base_path / dataset / dataset / scenario / str(run)
        
        # Load inject time (ground truth)
        inject_time = int((path / 'inject_time.txt').read_text().strip())
        
        # Load traces (RE2/RE3 have traces.csv, RE1 only has metrics)
        traces_file = path / 'traces.csv'
        if traces_file.exists():
            df = pd.read_csv(traces_file)
            df['is_anomaly'] = (df['startTimeMillis'] >= inject_time * 1000).astype(int)
            return df, inject_time
        return pd.DataFrame(), inject_time
    
    def load_metrics(self, dataset: str, scenario: str, run: int = 1) -> Tuple[pd.DataFrame, int]:
        """Load metrics data."""
        path = self.base_path / dataset / dataset / scenario / str(run)
        inject_time = int((path / 'inject_time.txt').read_text().strip())
        
        metrics_file = path / 'simple_metrics.csv' if (path / 'simple_metrics.csv').exists() else path / 'data.csv'
        df = pd.read_csv(metrics_file) if metrics_file.exists() else pd.DataFrame()
        return df, inject_time
    
    def list_scenarios(self, dataset: str) -> List[str]:
        """List all fault scenarios in a dataset."""
        path = self.base_path / dataset / dataset
        return [d.name for d in path.iterdir() if d.is_dir()] if path.exists() else []
    
    def get_fault_info(self, scenario: str) -> Dict[str, str]:
        """Parse fault info from scenario name."""
        parts = scenario.rsplit('_', 1)
        return {'service': parts[0], 'fault_type': parts[1] if len(parts) > 1 else 'unknown'}
    
    def to_trace_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert RCAEval traces to existing pipeline format."""
        if df.empty:
            return df
        return df.rename(columns={
            'traceID': 'trace_id',
            'spanID': 'span_id', 
            'serviceName': 'service_name',
            'startTimeMillis': 'start_time',
            'duration': 'duration_ms',
            'parentSpanID': 'parent_span_id'
        })[['trace_id', 'span_id', 'service_name', 'start_time', 'duration_ms', 'parent_span_id', 'is_anomaly']]
