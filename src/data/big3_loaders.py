"""
Unified Data Loader for Big 3 Production-Scale Datasets
- Meta Canopy (6.5M traces)
- Alibaba 2022 (20M+ traces)
- Uber Error Study (1.5M+ traces) - WITH LABELS
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator, Optional, Tuple
import json
import gzip

BASE_DIR = Path(__file__).parent.parent.parent / "data" / "raw"


class MetaCanopyLoader:
    """Load Meta's distributed traces (6.5M traces, 18.5K services)."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or BASE_DIR / "meta_canopy"
    
    def load_traces(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load trace data. Check repo for actual file structure after clone."""
        traces_dir = self.data_dir / "data"
        if not traces_dir.exists():
            raise FileNotFoundError(f"Meta data not found. Run: python scripts/download_big3_datasets.py --meta")
        
        # Meta format: JSON lines with trace_id, spans, services
        dfs = []
        for f in traces_dir.glob("*.json*"):
            df = pd.read_json(f, lines=True)
            dfs.append(df)
            if limit and sum(len(d) for d in dfs) >= limit:
                break
        
        return pd.concat(dfs, ignore_index=True).head(limit) if dfs else pd.DataFrame()
    
    def get_service_graph(self) -> dict:
        """Extract service dependency graph from traces."""
        # Implementation depends on actual Meta data format
        pass


class AlibabaLoader:
    """Load Alibaba cluster traces (20M+ traces, 20K services)."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or BASE_DIR / "alibaba_clusterdata"
    
    def load_call_graph(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load microservice call graph data."""
        # Alibaba 2022 format: MSCallGraph with columns:
        # timestamp, traceid, service, rpcid, um (upstream), dm (downstream), rt (response time)
        cg_dir = self.data_dir / "cluster-trace-microservices-v2022"
        
        if not cg_dir.exists():
            raise FileNotFoundError(f"Alibaba data not found. Run: python scripts/download_big3_datasets.py --alibaba")
        
        dfs = []
        for f in sorted(cg_dir.glob("MSCallGraph*.csv*")):
            df = pd.read_csv(f)
            dfs.append(df)
            if limit and sum(len(d) for d in dfs) >= limit:
                break
        
        return pd.concat(dfs, ignore_index=True).head(limit) if dfs else pd.DataFrame()
    
    def load_metrics(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load resource metrics (CPU, memory per container)."""
        metrics_dir = self.data_dir / "cluster-trace-microservices-v2022"
        
        dfs = []
        for f in sorted(metrics_dir.glob("MSResource*.csv*")):
            df = pd.read_csv(f)
            dfs.append(df)
            if limit and sum(len(d) for d in dfs) >= limit:
                break
        
        return pd.concat(dfs, ignore_index=True).head(limit) if dfs else pd.DataFrame()
    
    def iter_traces(self, batch_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Iterate through traces in batches (memory efficient)."""
        cg_dir = self.data_dir / "cluster-trace-microservices-v2022"
        
        for f in sorted(cg_dir.glob("MSCallGraph*.csv*")):
            for chunk in pd.read_csv(f, chunksize=batch_size):
                yield chunk


class UberErrorLoader:
    """Load Uber Error Study traces (1.5M+ traces, WITH ERROR LABELS)."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or BASE_DIR / "uber_error_study"
    
    def load_traces(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load traces with error labels preserved."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Uber data not found. Run: python scripts/download_big3_datasets.py --uber")
        
        # Uber format: JSON with spans containing error tags
        dfs = []
        for f in sorted(self.data_dir.glob("*.json*")):
            if f.suffix == ".gz":
                with gzip.open(f, "rt") as gz:
                    for line in gz:
                        dfs.append(json.loads(line))
                        if limit and len(dfs) >= limit:
                            break
            else:
                df = pd.read_json(f, lines=True)
                dfs.append(df)
            
            if limit and len(dfs) >= limit:
                break
        
        if isinstance(dfs[0], dict):
            return pd.DataFrame(dfs[:limit])
        return pd.concat(dfs, ignore_index=True).head(limit)
    
    def extract_errors(self, traces_df: pd.DataFrame) -> pd.DataFrame:
        """Extract error labels from traces."""
        # Uber preserves error tags in spans
        # Returns DataFrame with trace_id, has_error, error_type
        pass


class RCAEvalLoader:
    """Load RCAEval benchmark (735 labeled failure cases) - BACKUP."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or BASE_DIR / "rcaeval"
    
    def load_failures(self) -> pd.DataFrame:
        """Load failure cases with ground truth root causes."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"RCAEval not found. Run: python scripts/download_big3_datasets.py --rcaeval")
        
        # RCAEval has 3 systems: Online Boutique, Sock Shop, Train Ticket
        # Each with metrics, logs, traces and fault labels
        pass


def get_combined_stats() -> dict:
    """Get statistics across all loaded datasets."""
    stats = {
        "meta": {"traces": 0, "services": 0, "labels": False},
        "alibaba": {"traces": 0, "services": 0, "labels": False},
        "uber": {"traces": 0, "services": 0, "labels": True},
    }
    
    try:
        meta = MetaCanopyLoader()
        # Count from metadata if available
    except FileNotFoundError:
        pass
    
    try:
        alibaba = AlibabaLoader()
        # Count from metadata if available
    except FileNotFoundError:
        pass
    
    try:
        uber = UberErrorLoader()
        # Count from metadata if available
    except FileNotFoundError:
        pass
    
    return stats


if __name__ == "__main__":
    print("Big 3 Dataset Loaders")
    print("=" * 40)
    print("1. MetaCanopyLoader - 6.5M traces")
    print("2. AlibabaLoader - 20M+ traces")
    print("3. UberErrorLoader - 1.5M+ traces (LABELED)")
    print("4. RCAEvalLoader - 735 cases (BACKUP)")
