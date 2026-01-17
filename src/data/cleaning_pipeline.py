"""
Data Cleaning Pipeline
Implements OpenRefine-like operations for trace data cleaning.
Based on Library Carpentry OpenRefine guidelines.
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

from .data_validator import TidyDataValidator, validate_trace_dataframe

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CleaningOperation:
    """Represents a single cleaning operation."""
    name: str
    description: str
    column: Optional[str] = None
    function: Optional[Callable] = None
    params: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    rows_affected: int = 0


@dataclass
class CleaningLog:
    """Log of all cleaning operations performed."""
    operations: List[CleaningOperation] = field(default_factory=list)
    original_rows: int = 0
    final_rows: int = 0
    original_columns: int = 0
    final_columns: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert log to dictionary for export."""
        return {
            'original_rows': self.original_rows,
            'final_rows': self.final_rows,
            'rows_removed': self.original_rows - self.final_rows,
            'original_columns': self.original_columns,
            'final_columns': self.final_columns,
            'duration_seconds': (self.end_time - self.start_time).total_seconds()
                if self.start_time and self.end_time else 0,
            'operations': [
                {
                    'name': op.name,
                    'description': op.description,
                    'column': op.column,
                    'rows_affected': op.rows_affected
                }
                for op in self.operations
            ]
        }

    def save(self, filepath: str):
        """Save log to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class TraceDataCleaner:
    """
    Data cleaning pipeline for microservice trace data.
    Implements OpenRefine-like operations programmatically.
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Whether to print progress messages
        """
        self.verbose = verbose
        self.log = CleaningLog()
        self._df: Optional[pd.DataFrame] = None

    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load trace data from file.

        Args:
            filepath: Path to CSV or JSON file

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)

        if filepath.suffix == '.csv':
            self._df = pd.read_csv(filepath)
        elif filepath.suffix == '.json':
            self._df = pd.read_json(filepath)
        elif filepath.suffix == '.parquet':
            self._df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

        self.log.original_rows = len(self._df)
        self.log.original_columns = len(self._df.columns)
        self.log.start_time = datetime.now()

        self._log(f"Loaded {len(self._df)} rows, {len(self._df.columns)} columns from {filepath}")
        return self._df

    def _log(self, message: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            logger.info(message)

    def _record_operation(self, name: str, description: str,
                          column: Optional[str] = None,
                          rows_affected: int = 0):
        """Record a cleaning operation."""
        op = CleaningOperation(
            name=name,
            description=description,
            column=column,
            applied=True,
            rows_affected=rows_affected
        )
        self.log.operations.append(op)
        self._log(f"  {name}: {description} ({rows_affected} rows affected)")

    # ==========================================================================
    # PHASE 1: INITIAL INSPECTION (OpenRefine: Facet operations)
    # ==========================================================================

    def inspect(self) -> Dict:
        """
        Inspect DataFrame for data quality issues.

        Returns:
            Dictionary with inspection results
        """
        if self._df is None:
            raise RuntimeError("No data loaded. Call load() first.")

        df = self._df
        results = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }

        # Column-specific inspection
        results['column_stats'] = {}
        for col in df.columns:
            stats = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            }

            if df[col].dtype == 'object':
                # Text facet equivalent
                stats['sample_values'] = df[col].dropna().head(5).tolist()
                stats['min_length'] = df[col].dropna().str.len().min() if len(df[col].dropna()) > 0 else 0
                stats['max_length'] = df[col].dropna().str.len().max() if len(df[col].dropna()) > 0 else 0
            elif np.issubdtype(df[col].dtype, np.number):
                # Numeric facet equivalent
                stats['min'] = df[col].min()
                stats['max'] = df[col].max()
                stats['mean'] = df[col].mean()
                stats['std'] = df[col].std()
                stats['negative_count'] = (df[col] < 0).sum()
                stats['zero_count'] = (df[col] == 0).sum()

            results['column_stats'][col] = stats

        return results

    # ==========================================================================
    # PHASE 2: STANDARDIZATION (OpenRefine: Transform operations)
    # ==========================================================================

    def trim_whitespace(self, columns: Optional[List[str]] = None) -> 'TraceDataCleaner':
        """
        Trim whitespace from text columns.
        OpenRefine equivalent: Edit cells > Transform > value.trim()

        Args:
            columns: Columns to trim (None = all text columns)
        """
        if self._df is None:
            raise RuntimeError("No data loaded")

        if columns is None:
            columns = self._df.select_dtypes(include=['object']).columns.tolist()

        for col in columns:
            if col in self._df.columns:
                original = self._df[col].copy()
                self._df[col] = self._df[col].astype(str).str.strip()
                affected = (original != self._df[col]).sum()
                self._record_operation('trim_whitespace', f'Trimmed whitespace in {col}',
                                       col, affected)

        return self

    def standardize_case(self, columns: List[str],
                         case: str = 'lower') -> 'TraceDataCleaner':
        """
        Standardize case for text columns.
        OpenRefine equivalent: Edit cells > Transform > value.toLowercase()

        Args:
            columns: Columns to transform
            case: 'lower', 'upper', or 'title'
        """
        if self._df is None:
            raise RuntimeError("No data loaded")

        case_func = {'lower': str.lower, 'upper': str.upper, 'title': str.title}
        func = case_func.get(case, str.lower)

        for col in columns:
            if col in self._df.columns:
                original = self._df[col].copy()
                self._df[col] = self._df[col].astype(str).apply(
                    lambda x: func(x) if pd.notna(x) and x != 'nan' else x
                )
                affected = (original != self._df[col]).sum()
                self._record_operation('standardize_case',
                                       f'Applied {case} case to {col}',
                                       col, affected)

        return self

    def normalize_separators(self, columns: List[str],
                             target: str = '-') -> 'TraceDataCleaner':
        """
        Normalize separators (replace underscores, dots with target).
        OpenRefine equivalent: Edit cells > Transform > value.replace(...)

        Args:
            columns: Columns to transform
            target: Target separator character
        """
        if self._df is None:
            raise RuntimeError("No data loaded")

        pattern = r'[_\.\s]+'

        for col in columns:
            if col in self._df.columns:
                original = self._df[col].copy()
                self._df[col] = self._df[col].astype(str).str.replace(
                    pattern, target, regex=True
                )
                affected = (original != self._df[col]).sum()
                self._record_operation('normalize_separators',
                                       f'Replaced separators with "{target}" in {col}',
                                       col, affected)

        return self

    def standardize_service_names(self) -> 'TraceDataCleaner':
        """
        Comprehensive service name standardization.
        Combines: trim, lowercase, normalize separators.
        """
        if 'service_name' in self._df.columns:
            original = self._df['service_name'].copy()

            self._df['service_name'] = (
                self._df['service_name']
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r'[_\s]+', '-', regex=True)
                .str.replace(r'-+', '-', regex=True)
                .str.strip('-')
            )

            affected = (original != self._df['service_name']).sum()
            self._record_operation('standardize_service_names',
                                   'Applied comprehensive service name standardization',
                                   'service_name', affected)

        return self

    # ==========================================================================
    # PHASE 3: VALIDATION (OpenRefine: Facet + Filter operations)
    # ==========================================================================

    def handle_missing_values(self, strategy: str = 'drop',
                              columns: Optional[List[str]] = None,
                              fill_value: Any = None) -> 'TraceDataCleaner':
        """
        Handle missing values.
        OpenRefine equivalent: Facet > Text facet > Edit cells

        Args:
            strategy: 'drop', 'fill', or 'flag'
            columns: Columns to check (None = required columns)
            fill_value: Value to fill with (for 'fill' strategy)
        """
        if self._df is None:
            raise RuntimeError("No data loaded")

        if columns is None:
            columns = ['trace_id', 'span_id', 'service_name', 'duration_ms']

        existing_cols = [c for c in columns if c in self._df.columns]

        if strategy == 'drop':
            original_len = len(self._df)
            self._df = self._df.dropna(subset=existing_cols)
            affected = original_len - len(self._df)
            self._record_operation('handle_missing_values',
                                   f'Dropped rows with missing values in {existing_cols}',
                                   None, affected)

        elif strategy == 'fill':
            for col in existing_cols:
                null_count = self._df[col].isnull().sum()
                if null_count > 0:
                    fv = fill_value if fill_value is not None else (
                        0 if np.issubdtype(self._df[col].dtype, np.number) else ''
                    )
                    self._df[col] = self._df[col].fillna(fv)
                    self._record_operation('handle_missing_values',
                                           f'Filled {null_count} missing values in {col}',
                                           col, null_count)

        elif strategy == 'flag':
            self._df['has_missing'] = self._df[existing_cols].isnull().any(axis=1)
            missing_count = self._df['has_missing'].sum()
            self._record_operation('handle_missing_values',
                                   f'Flagged {missing_count} rows with missing values',
                                   None, missing_count)

        return self

    def validate_trace_ids(self, expected_length: int = 16) -> 'TraceDataCleaner':
        """
        Validate and standardize trace IDs.
        OpenRefine equivalent: Facet > Custom text facet > length(value)

        Args:
            expected_length: Expected length of trace IDs
        """
        if 'trace_id' not in self._df.columns:
            return self

        original = self._df['trace_id'].copy()

        # Standardize: lowercase, trim
        self._df['trace_id'] = (
            self._df['trace_id']
            .astype(str)
            .str.strip()
            .str.lower()
        )

        # Validate format
        valid_pattern = f'^[a-f0-9]{{{expected_length}}}$'
        invalid_mask = ~self._df['trace_id'].str.match(valid_pattern, na=False)
        invalid_count = invalid_mask.sum()

        if invalid_count > 0:
            self._log(f"  Warning: {invalid_count} invalid trace IDs found")
            # Flag invalid IDs
            self._df['invalid_trace_id'] = invalid_mask

        affected = (original != self._df['trace_id']).sum()
        self._record_operation('validate_trace_ids',
                               f'Validated trace IDs ({invalid_count} invalid)',
                               'trace_id', affected)

        return self

    def validate_durations(self, column: str = 'duration_ms',
                           remove_negative: bool = True) -> 'TraceDataCleaner':
        """
        Validate duration values.
        OpenRefine equivalent: Facet > Numeric facet

        Args:
            column: Duration column name
            remove_negative: Whether to remove rows with negative duration
        """
        if column not in self._df.columns:
            return self

        # Convert to numeric
        self._df[column] = pd.to_numeric(self._df[column], errors='coerce')

        # Handle negative values
        negative_mask = self._df[column] < 0
        negative_count = negative_mask.sum()

        if negative_count > 0:
            if remove_negative:
                self._df = self._df[~negative_mask]
                self._record_operation('validate_durations',
                                       f'Removed {negative_count} rows with negative duration',
                                       column, negative_count)
            else:
                self._df.loc[negative_mask, column] = 0
                self._record_operation('validate_durations',
                                       f'Set {negative_count} negative durations to 0',
                                       column, negative_count)

        return self

    def remove_duplicates(self, subset: Optional[List[str]] = None) -> 'TraceDataCleaner':
        """
        Remove duplicate rows.
        OpenRefine equivalent: Sort > Edit rows > Remove matching rows

        Args:
            subset: Columns to consider for duplicates (None = all)
        """
        if self._df is None:
            raise RuntimeError("No data loaded")

        if subset is None:
            subset = ['trace_id', 'span_id']
        subset = [c for c in subset if c in self._df.columns]

        original_len = len(self._df)
        self._df = self._df.drop_duplicates(subset=subset, keep='first')
        removed = original_len - len(self._df)

        self._record_operation('remove_duplicates',
                               f'Removed {removed} duplicate rows based on {subset}',
                               None, removed)

        return self

    # ==========================================================================
    # PHASE 4: TRANSFORMATION (OpenRefine: Add column, Extract, Split)
    # ==========================================================================

    def parse_duration_with_units(self, column: str = 'duration',
                                   target_column: str = 'duration_ms') -> 'TraceDataCleaner':
        """
        Parse duration values that include units (e.g., "150ms", "0.5s").
        OpenRefine equivalent: Edit cells > Transform

        Args:
            column: Source column with duration values
            target_column: Target column for parsed values
        """
        if column not in self._df.columns:
            return self

        def parse_duration(val):
            if pd.isna(val):
                return np.nan
            val = str(val).strip().lower()

            # Already numeric
            try:
                return float(val)
            except ValueError:
                pass

            # Parse with units
            if 'ms' in val:
                return float(val.replace('ms', '').strip())
            elif 's' in val:
                return float(val.replace('s', '').strip()) * 1000
            elif 'us' in val or 'µs' in val:
                return float(re.sub(r'[uµ]s', '', val).strip()) / 1000

            return np.nan

        self._df[target_column] = self._df[column].apply(parse_duration)
        affected = self._df[target_column].notna().sum()

        self._record_operation('parse_duration_with_units',
                               f'Parsed durations from {column} to {target_column}',
                               column, affected)

        return self

    def parse_timestamps(self, column: str = 'start_time',
                         target_format: str = 'microseconds') -> 'TraceDataCleaner':
        """
        Parse various timestamp formats to Unix microseconds.
        OpenRefine equivalent: Edit cells > Transform > toDate()

        Args:
            column: Timestamp column
            target_format: 'microseconds', 'milliseconds', or 'seconds'
        """
        if column not in self._df.columns:
            return self

        def parse_ts(val):
            if pd.isna(val):
                return np.nan

            # Already numeric (assume microseconds)
            if isinstance(val, (int, float)):
                return val

            val = str(val).strip()

            # ISO 8601 format
            if 'T' in val:
                try:
                    dt = pd.to_datetime(val)
                    return int(dt.timestamp() * 1e6)
                except:
                    pass

            # Unix timestamp (various formats)
            try:
                num_val = float(val)
                # Heuristic: if very large, it's already microseconds
                if num_val > 1e15:
                    return num_val
                elif num_val > 1e12:
                    return num_val * 1000  # Milliseconds to microseconds
                else:
                    return num_val * 1e6  # Seconds to microseconds
            except:
                pass

            return np.nan

        self._df[column] = self._df[column].apply(parse_ts)
        affected = self._df[column].notna().sum()

        self._record_operation('parse_timestamps',
                               f'Parsed timestamps in {column}',
                               column, affected)

        return self

    def split_tags(self, column: str = 'tags',
                   separator: str = ',') -> 'TraceDataCleaner':
        """
        Split combined tag column into separate columns.
        OpenRefine equivalent: Edit column > Split into several columns

        Args:
            column: Column with combined tags
            separator: Tag separator
        """
        if column not in self._df.columns:
            return self

        def parse_tags(val):
            if pd.isna(val):
                return {}
            try:
                # Try JSON first
                if isinstance(val, str) and val.startswith('{'):
                    return json.loads(val)
                # Key:value pairs
                pairs = str(val).split(separator)
                result = {}
                for pair in pairs:
                    if ':' in pair:
                        key, value = pair.split(':', 1)
                        result[key.strip()] = value.strip()
                return result
            except:
                return {}

        tags_df = self._df[column].apply(parse_tags).apply(pd.Series)
        new_cols = [f'tag_{c}' for c in tags_df.columns]
        tags_df.columns = new_cols

        # Add to main DataFrame
        for col in new_cols:
            if col not in self._df.columns:
                self._df[col] = tags_df[col]

        self._record_operation('split_tags',
                               f'Split {column} into {len(new_cols)} columns',
                               column, len(self._df))

        return self

    def add_derived_column(self, name: str,
                           expression: Callable[[pd.DataFrame], pd.Series]) -> 'TraceDataCleaner':
        """
        Add a derived column based on an expression.
        OpenRefine equivalent: Edit column > Add column based on this column

        Args:
            name: New column name
            expression: Function that takes DataFrame and returns Series
        """
        if self._df is None:
            raise RuntimeError("No data loaded")

        self._df[name] = expression(self._df)

        self._record_operation('add_derived_column',
                               f'Added derived column: {name}',
                               name, len(self._df))

        return self

    # ==========================================================================
    # PHASE 5: EXPORT
    # ==========================================================================

    def finalize(self) -> pd.DataFrame:
        """
        Finalize cleaning and return cleaned DataFrame.

        Returns:
            Cleaned DataFrame
        """
        if self._df is None:
            raise RuntimeError("No data loaded")

        self.log.final_rows = len(self._df)
        self.log.final_columns = len(self._df.columns)
        self.log.end_time = datetime.now()

        # Validate with TidyDataValidator
        self._log("\nValidating cleaned data...")
        validator = TidyDataValidator(
            required_columns=['trace_id', 'span_id', 'service_name', 'duration_ms']
        )
        results = validator.validate(self._df)
        validator.print_report()

        self._log(f"\nCleaning complete: {self.log.original_rows} -> {self.log.final_rows} rows")
        self._log(f"Total operations: {len(self.log.operations)}")

        return self._df

    def save(self, filepath: str, save_log: bool = True):
        """
        Save cleaned data and cleaning log.

        Args:
            filepath: Output file path
            save_log: Whether to save cleaning log
        """
        if self._df is None:
            raise RuntimeError("No data to save")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save data
        if filepath.suffix == '.csv':
            self._df.to_csv(filepath, index=False)
        elif filepath.suffix == '.parquet':
            self._df.to_parquet(filepath, index=False)
        elif filepath.suffix == '.json':
            self._df.to_json(filepath, orient='records', indent=2)

        self._log(f"Saved cleaned data to: {filepath}")

        # Save log
        if save_log:
            log_path = filepath.with_suffix('.cleaning_log.json')
            self.log.save(str(log_path))
            self._log(f"Saved cleaning log to: {log_path}")


def clean_trace_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Convenience function to clean a trace dataset with default settings.

    Args:
        input_path: Path to input file
        output_path: Path to save cleaned data

    Returns:
        Cleaned DataFrame
    """
    cleaner = TraceDataCleaner(verbose=True)

    df = (cleaner
          .load(input_path)
          .trim_whitespace()
          .standardize_service_names()
          .handle_missing_values(strategy='drop')
          .validate_trace_ids()
          .validate_durations()
          .remove_duplicates()
          .finalize())

    cleaner.save(output_path, save_log=True)

    return df


# Example usage
if __name__ == "__main__":
    # Example: Clean a trace dataset
    import sys

    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        clean_trace_dataset(input_file, output_file)
    else:
        print("Usage: python cleaning_pipeline.py <input_file> <output_file>")
        print("\nExample:")
        print("  python cleaning_pipeline.py data/raw/traces.csv data/processed/traces_clean.csv")
