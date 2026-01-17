"""S3-based RCAEval data loader for direct cloud processing."""
import pandas as pd
import boto3
import io
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class S3RCAEvalLoader:
    """Load RCAEval datasets directly from S3."""
    
    def __init__(self, bucket: str = "cse713-research-datasets", prefix: str = "rcaeval"):
        self.bucket = bucket
        self.prefix = prefix
        self.s3 = boto3.client('s3')
    
    def load_traces(self, dataset: str, scenario: str, run: int = 1) -> Tuple[pd.DataFrame, int]:
        """Load traces from S3.
        
        Args:
            dataset: 'RE1-TT', 'RE2-TT', or 'RE3-TT'
            scenario: e.g., 'ts-auth-service_cpu'
            run: Run number (1-5)
        
        Returns:
            (traces_df, inject_time)
        """
        path = f"{self.prefix}/{dataset}/{dataset}/{scenario}/{run}"
        
        # Load inject_time
        inject_obj = self.s3.get_object(Bucket=self.bucket, Key=f"{path}/inject_time.txt")
        inject_time = int(inject_obj['Body'].read().decode().strip())
        
        # Load traces if exists
        try:
            traces_obj = self.s3.get_object(Bucket=self.bucket, Key=f"{path}/traces.csv")
            df = pd.read_csv(io.BytesIO(traces_obj['Body'].read()))
            df['is_anomaly'] = (df['startTimeMillis'] >= inject_time * 1000).astype(int)
            return df, inject_time
        except:
            return pd.DataFrame(), inject_time
    
    def load_metrics(self, dataset: str, scenario: str, run: int = 1) -> Tuple[pd.DataFrame, int]:
        """Load metrics from S3."""
        path = f"{self.prefix}/{dataset}/{dataset}/{scenario}/{run}"
        
        inject_obj = self.s3.get_object(Bucket=self.bucket, Key=f"{path}/inject_time.txt")
        inject_time = int(inject_obj['Body'].read().decode().strip())
        
        # Try simple_metrics first, then data.csv
        for fname in ['simple_metrics.csv', 'data.csv', 'metrics.csv']:
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=f"{path}/{fname}")
                df = pd.read_csv(io.BytesIO(obj['Body'].read()))
                return df, inject_time
            except:
                continue
        
        return pd.DataFrame(), inject_time
    
    def list_scenarios(self, dataset: str) -> List[str]:
        """List all scenarios in dataset."""
        prefix = f"{self.prefix}/{dataset}/{dataset}/"
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, Delimiter='/')
        
        scenarios = []
        if 'CommonPrefixes' in response:
            for p in response['CommonPrefixes']:
                scenario = p['Prefix'].split('/')[-2]
                if scenario:
                    scenarios.append(scenario)
        return sorted(scenarios)
    
    def get_fault_info(self, scenario: str) -> Dict[str, str]:
        """Parse fault info from scenario name."""
        parts = scenario.rsplit('_', 1)
        return {'service': parts[0], 'fault_type': parts[1] if len(parts) > 1 else 'unknown'}
    
    def to_trace_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert RCAEval traces to pipeline format."""
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
