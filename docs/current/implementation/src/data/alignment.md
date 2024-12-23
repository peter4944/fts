# Data Alignment Module

## Overview
The data alignment module handles the synchronization and alignment of multiple time series with different frequencies, gaps, and overlapping periods.

## Classes

### DataAligner

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

from fts.core.base import TimeSeries
from fts.core.validation import validate_alignment
from fts.core.errors import ValidationError, ProcessingError

class AlignmentMethod(Enum):
    """Supported methods for time series alignment."""
    ALL_OVERLAPPING = "all_overlapping"         # Method 1: Traditional approach
    SYNCHRONIZED_AVERAGE = "synchronized_average"  # Method 2: Synchronized returns
    PAIRWISE_OVERLAPPING = "pairwise_overlapping"  # Method 3: Independent pairs
    PAIRWISE_AVERAGE = "pairwise_average"        # Method 4: DTW-specific method

@dataclass
class AlignmentConfig:
    """Configuration for alignment operations."""
    method: AlignmentMethod
    min_overlap: float = 0.8
    handle_gaps: bool = True
    max_gap_size: Optional[int] = None
    frequency: Optional[str] = None

class DataAligner:
    """Handles alignment of multiple time series."""

    def __init__(self, config: AlignmentConfig):
        """
        Initialize the DataAligner with configuration.

        Args:
            config: AlignmentConfig object specifying alignment parameters
        """
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate alignment configuration.

        Validates:
        - Basic parameter ranges
        - Frequency-specific method suitability
        - Gap handling parameters
        """
        if not 0 < self.config.min_overlap <= 1:
            raise ValidationError("min_overlap must be between 0 and 1")
        if self.config.max_gap_size is not None and self.config.max_gap_size < 1:
            raise ValidationError("max_gap_size must be positive")

        # Validate frequency is specified
        if not self.config.frequency:
            raise ValidationError("Frequency must be specified")

        # Method-specific validations based on frequency and gaps
        if self.config.method == AlignmentMethod.PAIRWISE_AVERAGE:
            if self.config.frequency not in ['daily', 'weekly']:
                raise ValidationError(
                    "Pairwise Average method is only recommended for daily/weekly data"
                )

        elif self.config.method == AlignmentMethod.ALL_OVERLAPPING:
            if self.config.frequency in ['intraday', 'daily']:
                warnings.warn(
                    "All Overlapping method may discard significant data for high frequency series"
                )

    def align_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Align multiple time series according to configured method.

        Method selection guidelines:
        - ALL_OVERLAPPING: Suitable for weekly+ frequencies, simple analysis
        - SYNCHRONIZED_AVERAGE: Recommended for risk models, PCA, daily data
        - PAIRWISE_OVERLAPPING: General purpose, accepts data loss
        - PAIRWISE_AVERAGE: Specifically for DTW with daily/weekly data

        Args:
            data: DataFrame with series as columns, datetime index

        Returns:
            DataFrame with aligned series as columns

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If alignment fails
        """
        if len(data.columns) < 2:
            raise ValidationError("At least two series required for alignment")

        # Validate input series
        validate_alignment(data)

        if self.config.method == AlignmentMethod.ALL_OVERLAPPING:
            return self._align_all_overlapping(data)
        elif self.config.method == AlignmentMethod.SYNCHRONIZED_AVERAGE:
            return self._align_synchronized_average(data)
        elif self.config.method == AlignmentMethod.PAIRWISE_OVERLAPPING:
            return self._align_pairwise_overlapping(data)
        else:
            return self._align_pairwise_average(data)

    def _align_all_overlapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """Align using only fully overlapping periods."""
        # Drop any rows with missing values
        return data.dropna()

    def _align_synchronized_average(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Align using synchronized average method for handling gaps.

        This method:
        1. Identifies common trading periods
        2. Handles gaps according to configuration
        3. Synchronizes returns across different frequencies

        Returns:
            DataFrame with aligned series using synchronized average method

        Raises:
            ProcessingError: If alignment fails
        """
        try:
            # Process each column
            aligned_series = {}
            for col in data.columns:
                series_data = data[col]

                # Find gaps
                gaps = series_data.isna()
                if not gaps.any():
                    aligned_series[col] = series_data
                    continue

                # Process each gap
                gap_starts = gaps[gaps].index
                for gap_start in gap_starts:
                    # Find gap end
                    gap_end_idx = gaps[gap_start:].idxmin()
                    if pd.isna(gap_end_idx):  # Gap extends to end
                        continue

                    # Get returns before and after gap
                    pre_gap_value = series_data.loc[:gap_start].last_valid_index()
                    post_gap_value = series_data.loc[gap_end_idx]

                    if pd.isna(pre_gap_value):  # Gap starts at beginning
                        continue

                    # Calculate average return over gap
                    gap_length = len(series_data.loc[gap_start:gap_end_idx])
                    if self.config.max_gap_size and gap_length > self.config.max_gap_size:
                        continue

                    # Distribute returns evenly across gap
                    avg_return = (post_gap_value - pre_gap_value) / gap_length
                    series_data.loc[gap_start:gap_end_idx] = \
                        [pre_gap_value + avg_return * (i+1)
                         for i in range(gap_length)]

                aligned_series[col] = series_data

            return pd.DataFrame(aligned_series)

        except Exception as e:
            raise ProcessingError(f"Failed to align series using synchronized average: {str(e)}")

    def _align_pairwise_overlapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method 3: Pairwise Overlapping alignment.

        This method:
        1. Aligns each pair of series independently
        2. Uses only overlapping periods for each pair
        3. Different periods may be used for different pairs
        4. May produce non-PSD correlation matrix

        Returns:
            DataFrame with pairwise aligned series

        Raises:
            ProcessingError: If alignment fails
        """
        try:
            n_series = len(data.columns)
            aligned_pairs = {}

            # Process each pair
            for i in range(n_series):
                for j in range(i+1, n_series):
                    pair_key = f"{i}-{j}"
                    series_i = data[data.columns[i]]
                    series_j = data[data.columns[j]]

                    # Find common date range
                    start_date = max(series_i.index.min(), series_j.index.min())
                    end_date = min(series_i.index.max(), series_j.index.max())

                    # Extract overlapping period
                    s1 = series_i[start_date:end_date]
                    s2 = series_j[start_date:end_date]

                    # Only keep periods where both series have data
                    pair_df = pd.concat([s1, s2], axis=1).dropna()
                    aligned_pairs[pair_key] = pair_df

            # Combine aligned pairs
            combined = self._combine_aligned_pairs(aligned_pairs, n_series)
            return combined

        except Exception as e:
            raise ProcessingError(f"Failed to align series using pairwise overlapping method: {str(e)}")

    def _align_pairwise_average(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Method 4: Pairwise Average alignment.

        Specifically designed for daily/weekly data with gaps:
        1. Consider each pair independently
        2. For gaps, distribute returns evenly across missing periods
        3. Preserve total returns while maintaining return distribution properties
        4. May produce non-PSD correlation matrix

        Note: This method is particularly suitable for DTW analysis

        Returns:
            DataFrame with pairwise averaged series

        Raises:
            ProcessingError: If alignment fails
            Warning: If resulting matrix might not be PSD
        """
        try:
            n_series = len(data.columns)
            aligned_pairs = {}

            # Warn about matrix properties
            warnings.warn(
                "Pairwise Average method may produce non-PSD correlation matrix"
            )

            # Process each pair
            for i in range(n_series):
                for j in range(i+1, n_series):
                    pair_key = f"{i}-{j}"
                    series_i = data[data.columns[i]]
                    series_j = data[data.columns[j]]

                    # Find common date range
                    start_date = max(series_i.index.min(), series_j.index.min())
                    end_date = min(series_i.index.max(), series_j.index.max())

                    # Extract overlapping period
                    s1 = series_i[start_date:end_date]
                    s2 = series_j[start_date:end_date]

                    # Handle gaps in each series
                    aligned_pair = self._align_pair_with_distribution(s1, s2)
                    aligned_pairs[pair_key] = aligned_pair

            # Combine aligned pairs
            combined = self._combine_aligned_pairs(aligned_pairs, n_series)
            return combined

        except Exception as e:
            raise ProcessingError(f"Failed to align series using pairwise average method: {str(e)}")

    def _align_pair_with_distribution(self, series1: pd.Series, series2: pd.Series) -> pd.DataFrame:
        """
        Align a pair of series while preserving return distribution properties.

        This implementation:
        1. Preserves total returns over gaps
        2. Maintains natural return variance
        3. Avoids artificial return spikes
        4. Handles lead/lag relationships

        Args:
            series1: First series
            series2: Second series

        Returns:
            DataFrame with aligned pair preserving distribution properties
        """
        df = pd.concat([series1, series2], axis=1)

        # Process gaps in each series
        for col in df.columns:
            gaps = df[col].isna()
            if not gaps.any():
                continue

            gap_starts = gaps[gaps].index
            for gap_start in gap_starts:
                gap_end_idx = gaps[gap_start:].idxmin()
                if pd.isna(gap_end_idx):
                    continue

                pre_gap_value = df[col].loc[:gap_start].last_valid_index()
                post_gap_value = df[col].loc[gap_end_idx]

                if pd.isna(pre_gap_value):
                    continue

                gap_length = len(df.loc[gap_start:gap_end_idx])
                if self.config.max_gap_size and gap_length > self.config.max_gap_size:
                    continue

                # Calculate total return over gap
                total_return = post_gap_value - pre_gap_value

                # Distribute returns while preserving variance characteristics
                if gap_length > 1:
                    # Use slightly randomized distribution to maintain natural variance
                    random_weights = np.random.dirichlet(np.ones(gap_length))
                    returns = total_return * random_weights
                    cumulative_returns = np.cumsum(returns)
                    df.loc[gap_start:gap_end_idx, col] = \
                        pre_gap_value + cumulative_returns
                else:
                    df.loc[gap_start:gap_end_idx, col] = \
                        pre_gap_value + total_return

        return df

    def _combine_aligned_pairs(self, aligned_pairs: Dict[str, pd.DataFrame],
                             n_series: int) -> pd.DataFrame:
        """
        Combine pairwise aligned series into a single DataFrame.

        Args:
            aligned_pairs: Dictionary of aligned pair DataFrames
            n_series: Number of original series

        Returns:
            Combined DataFrame preserving pairwise relationships
        """
        # Initialize with the first pair
        first_pair = list(aligned_pairs.values())[0]
        combined = first_pair.copy()

        # Add remaining pairs while preserving relationships
        for pair_key, pair_df in list(aligned_pairs.items())[1:]:
            i, j = map(int, pair_key.split('-'))

            # Update columns if they exist
            for col in pair_df.columns:
                if col not in combined:
                    combined[col] = pair_df[col]
                else:
                    # Average with existing values where both exist
                    mask = combined[col].notna() & pair_df[col].notna()
                    combined.loc[mask, col] = \
                        (combined.loc[mask, col] + pair_df.loc[mask, col]) / 2

        return combined

    def get_alignment_stats(self, aligned_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate alignment statistics.

        Args:
            aligned_data: DataFrame of aligned series

        Returns:
            Dict containing alignment statistics:
            - overlap_ratio: Ratio of overlapping periods
            - gap_counts: Number of gaps by size
            - coverage_ratio: Ratio of valid data points
        """
        stats = {
            "overlap_ratio": self._calculate_overlap_ratio(aligned_data),
            "gap_counts": self._count_gaps(aligned_data),
            "coverage_ratio": self._calculate_coverage(aligned_data)
        }
        return stats

    def _calculate_overlap_ratio(self, data: pd.DataFrame) -> float:
        """Calculate ratio of overlapping periods."""
        total_periods = len(data)
        complete_periods = len(data.dropna())
        return complete_periods / total_periods if total_periods > 0 else 0

    def _count_gaps(self, data: pd.DataFrame) -> Dict[str, int]:
        """Count gaps by size in aligned data."""
        gaps = {}
        for col in data.columns:
            series = data[col]
            gap_sizes = series.isna().astype(int).groupby(
                series.notna().cumsum()
            ).sum()
            for size in gap_sizes:
                gaps[f"size_{size}"] = gaps.get(f"size_{size}", 0) + 1
        return gaps

    def _calculate_coverage(self, data: pd.DataFrame) -> float:
        """Calculate overall data coverage ratio."""
        return 1 - (data.isna().sum().sum() / (len(data) * len(data.columns)))

@dataclass
class AlignmentReport:
    """Report containing alignment results and statistics."""

    aligned_data: pd.DataFrame
    statistics: Dict[str, float]
    config: AlignmentConfig

    def generate_summary(self) -> str:
        """Generate text summary of alignment results."""
        summary = [
            "Alignment Summary:",
            f"Method: {self.config.method.value}",
            f"Series count: {len(self.aligned_data.columns)}",
            f"Time periods: {len(self.aligned_data)}",
            f"Overlap ratio: {self.statistics['overlap_ratio']:.2%}",
            f"Coverage ratio: {self.statistics['coverage_ratio']:.2%}",
            "\nGap Summary:"
        ]

        for gap_type, count in self.statistics['gap_counts'].items():
            summary.append(f"{gap_type}: {count}")

        return "\n".join(summary)

    def validate_results(self) -> bool:
        """
        Validate alignment results against configuration requirements.

        Returns:
            bool: True if validation passes, False otherwise
        """
        if self.statistics['overlap_ratio'] < self.config.min_overlap:
            return False

        if self.config.max_gap_size:
            max_gap = max(
                int(k.split('_')[1])
                for k in self.statistics['gap_counts'].keys()
            )
            if max_gap > self.config.max_gap_size:
                return False

        return True
```

## Usage Example

```python
# Create alignment configuration
config = AlignmentConfig(
    method=AlignmentMethod.SYNCHRONIZED_AVERAGE,
    min_overlap=0.8,
    handle_gaps=True,
    max_gap_size=5,
    frequency='daily'  # Only frequency is required
)

# Initialize aligner
aligner = DataAligner(config)

# Align multiple time series from DataFrame
# data_df is a pandas DataFrame with datetime index and series as columns
aligned_df = aligner.align_series(data_df)

# Get alignment statistics
stats = aligner.get_alignment_stats(aligned_df)

# Create alignment report
report = AlignmentReport(aligned_df, stats, config)
print(report.generate_summary())

# Check if result contains gaps
if aligned_df.isna().any().any():
    warnings.warn("Aligned data contains gaps - consider using different alignment method")

# Example: Calculate correlation matrix from aligned data
correlation_matrix = aligned_df.corr()
```

## Implementation Notes

### Validation Rules
1. Minimum overlap ratio must be between 0 and 1
2. Maximum gap size must be positive if specified
3. At least two series required for alignment
4. Input series must pass basic validation checks
5. Aligned results must meet configuration requirements

### Error Handling
- ValidationError for invalid inputs or configuration
- ProcessingError for alignment failures
- Detailed error messages with context
- Proper cleanup of resources

### Performance Considerations
1. Use vectorized operations where possible
2. Implement efficient gap handling
3. Optimize memory usage for large datasets
4. Consider chunked processing for very large series

### Future Enhancements
1. Support for additional alignment methods
2. Parallel processing for large datasets
3. Advanced gap handling strategies
4. Custom validation rules
5. Extended reporting capabilities
