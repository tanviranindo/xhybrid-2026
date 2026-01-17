"""S3-based data processing pipeline for RCAEval datasets."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from .s3_loader import S3RCAEvalLoader

class S3DataPipeline:
    """Process RCAEval data from S3 for model training."""
    
    def __init__(self, bucket: str = "cse713-research-datasets"):
        self.loader = S3RCAEvalLoader(bucket=bucket)
        self.datasets = ['RE1-TT', 'RE2-TT', 'RE3-TT']
    
    def process_dataset(self, dataset: str, max_scenarios: Optional[int] = None) -> Dict:
        """Process all scenarios in a dataset.
        
        Returns:
            {
                'traces': list of DataFrames,
                'metrics': list of DataFrames,
                'labels': list of labels,
                'scenarios': list of scenario names
            }
        """
        scenarios = self.loader.list_scenarios(dataset)
        if max_scenarios:
            scenarios = scenarios[:max_scenarios]
        
        results = {
            'traces': [],
            'metrics': [],
            'labels': [],
            'scenarios': [],
            'fault_types': []
        }
        
        for scenario in scenarios:
            fault_info = self.loader.get_fault_info(scenario)
            
            # Load run 1 (representative)
            traces_df, inject_time = self.loader.load_traces(dataset, scenario, run=1)
            metrics_df, _ = self.loader.load_metrics(dataset, scenario, run=1)
            
            if not traces_df.empty:
                traces_df = self.loader.to_trace_format(traces_df)
                results['traces'].append(traces_df)
                results['labels'].append(traces_df['is_anomaly'].values)
                results['scenarios'].append(scenario)
                results['fault_types'].append(fault_info['fault_type'])
            
            if not metrics_df.empty:
                results['metrics'].append(metrics_df)
        
        return results
    
    def get_dataset_stats(self) -> Dict:
        """Get statistics for all datasets."""
        stats = {}
        for dataset in self.datasets:
            scenarios = self.loader.list_scenarios(dataset)
            stats[dataset] = {
                'scenarios': len(scenarios),
                'fault_types': len(set(self.loader.get_fault_info(s)['fault_type'] for s in scenarios))
            }
        return stats
