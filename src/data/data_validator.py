"""
Data Validation Module
Validates trace data against tidy data principles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class TidyDataValidator:
    """
    Validates DataFrames against tidy data principles.
    
    Based on Library Carpentry tidy data guidelines:
    https://librarycarpentry.github.io/lc-spreadsheets/
    """
    
    def __init__(self, required_columns: Optional[List[str]] = None):
        """
        Args:
            required_columns: List of required column names
        """
        self.required_columns = required_columns or []
        self.validation_results = {}
    
    def validate(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Run all validation checks on DataFrame.
        
        Returns:
            Dictionary of check names and results (True = pass, False = fail)
        """
        results = {}
        
        # Structure checks
        results['has_required_columns'] = self._check_required_columns(df)
        results['no_duplicate_rows'] = self._check_duplicate_rows(df)
        results['no_empty_cells_in_required'] = self._check_empty_cells(df)
        
        # Data type checks
        results['consistent_data_types'] = self._check_data_types(df)
        
        # Value validation
        results['valid_trace_ids'] = self._check_trace_ids(df) if 'trace_id' in df.columns else True
        results['non_negative_duration'] = self._check_duration(df) if 'duration_ms' in df.columns else True
        results['valid_labels'] = self._check_labels(df) if 'is_anomaly' in df.columns else True
        
        # Tidy data principles
        results['one_value_per_cell'] = self._check_one_value_per_cell(df)
        results['no_merged_columns'] = self._check_no_merged_columns(df)
        
        self.validation_results = results
        return results
    
    def _check_required_columns(self, df: pd.DataFrame) -> bool:
        """Check if all required columns are present."""
        if not self.required_columns:
            return True
        missing = set(self.required_columns) - set(df.columns)
        if missing:
            print(f"⚠️  Missing required columns: {missing}")
            return False
        return True
    
    def _check_duplicate_rows(self, df: pd.DataFrame) -> bool:
        """Check for duplicate rows."""
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"⚠️  Found {duplicates} duplicate rows")
            return False
        return True
    
    def _check_empty_cells(self, df: pd.DataFrame) -> bool:
        """Check for empty cells in required columns."""
        if not self.required_columns:
            return True
        
        empty_counts = df[self.required_columns].isna().sum()
        has_empty = empty_counts.sum() > 0
        
        if has_empty:
            print("⚠️  Empty cells in required columns:")
            for col, count in empty_counts.items():
                if count > 0:
                    print(f"   - {col}: {count} empty cells")
            return False
        return True
    
    def _check_data_types(self, df: pd.DataFrame) -> bool:
        """Check for consistent data types within columns."""
        issues = []
        for col in df.columns:
            # Check for mixed types (object columns with mixed types)
            if df[col].dtype == 'object':
                # Try to detect mixed types
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    types = non_null.apply(type).unique()
                    if len(types) > 1:
                        issues.append(f"{col}: mixed types {types}")
        
        if issues:
            print("⚠️  Data type inconsistencies:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        return True
    
    def _check_trace_ids(self, df: pd.DataFrame) -> bool:
        """Validate trace IDs (should be unique, non-empty strings)."""
        trace_ids = df['trace_id']
        
        # Check for empty/null
        if trace_ids.isna().any():
            print("⚠️  Found null/empty trace_ids")
            return False
        
        # Check for duplicates (if this is span-level data, duplicates are OK)
        # But trace_ids should still be non-empty
        if (trace_ids == '').any():
            print("⚠️  Found empty string trace_ids")
            return False
        
        return True
    
    def _check_duration(self, df: pd.DataFrame) -> bool:
        """Check duration values are non-negative."""
        durations = df['duration_ms']
        
        if durations.isna().any():
            print("⚠️  Found null duration values")
            return False
        
        if (durations < 0).any():
            negative_count = (durations < 0).sum()
            print(f"⚠️  Found {negative_count} negative duration values")
            return False
        
        return True
    
    def _check_labels(self, df: pd.DataFrame) -> bool:
        """Check labels are valid (binary: 0 or 1)."""
        labels = df['is_anomaly']
        
        if labels.isna().any():
            print("⚠️  Found null labels")
            return False
        
        valid_values = labels.isin([0, 1])
        if not valid_values.all():
            invalid = labels[~valid_values].unique()
            print(f"⚠️  Found invalid label values: {invalid}")
            return False
        
        return True
    
    def _check_one_value_per_cell(self, df: pd.DataFrame) -> bool:
        """
        Check that cells don't contain multiple values.
        This is a heuristic check - looks for common separators.
        """
        issues = []
        separators = [',', ';', '|', '/', '\\t']
        
        for col in df.select_dtypes(include=['object']).columns:
            for sep in separators:
                # Check if any cell contains the separator
                contains_sep = df[col].astype(str).str.contains(sep, na=False, regex=False)
                if contains_sep.any():
                    count = contains_sep.sum()
                    issues.append(f"{col}: {count} cells contain '{sep}' (possible multi-value)")
                    break  # Only report once per column
        
        if issues:
            print("⚠️  Possible multi-value cells detected:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        
        return True
    
    def _check_no_merged_columns(self, df: pd.DataFrame) -> bool:
        """
        Check for signs of merged columns (heuristic).
        This is hard to detect programmatically, but we can check for
        columns that are always empty or have identical values.
        """
        # Check for columns that are completely empty
        empty_cols = df.columns[df.isna().all()].tolist()
        if empty_cols:
            print(f"⚠️  Found completely empty columns (possible merged cells): {empty_cols}")
            return False
        
        return True
    
    def get_report(self) -> str:
        """Generate a validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate() first."
        
        report = ["=" * 60]
        report.append("TIDY DATA VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        passed = sum(1 for v in self.validation_results.values() if v)
        total = len(self.validation_results)
        
        report.append(f"Results: {passed}/{total} checks passed")
        report.append("")
        
        for check, result in self.validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            report.append(f"{status}: {check}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def print_report(self):
        """Print validation report."""
        print(self.get_report())


def validate_trace_dataframe(df: pd.DataFrame, 
                             required_columns: Optional[List[str]] = None) -> Tuple[bool, Dict[str, bool]]:
    """
    Convenience function to validate a trace DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: Optional list of required columns
    
    Returns:
        (all_passed, results_dict)
    """
    if required_columns is None:
        required_columns = ['trace_id', 'span_id', 'service_name', 'duration_ms']
    
    validator = TidyDataValidator(required_columns=required_columns)
    results = validator.validate(df)
    all_passed = all(results.values())
    
    validator.print_report()
    
    return all_passed, results


# Example usage
if __name__ == "__main__":
    # Create sample data
    sample_data = {
        'trace_id': ['t1', 't1', 't2', 't2'],
        'span_id': ['s1', 's2', 's3', 's4'],
        'service_name': ['service-a', 'service-b', 'service-a', 'service-c'],
        'duration_ms': [100.5, 200.0, 150.0, 300.0],
        'is_anomaly': [0, 0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Validate
    validator = TidyDataValidator(required_columns=['trace_id', 'span_id', 'service_name', 'duration_ms'])
    results = validator.validate(df)
    validator.print_report()
