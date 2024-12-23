# DTW Correlation Module

## Overview
This module handles the conversion of DTW similarities to correlation coefficients.

## Classes

### DTWCorrelation

```python
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import warnings

from fts.core.errors import ValidationError, ProcessingError
from fts.dtw.similarity import DTWSimilarity
from fts.data.alignment import DataAligner, AlignmentConfig, AlignmentMethod
from fts.statistics.matrix.operations import (
    SimilarityMatrix,
    CorrelationCovarianceMatrix
)
from fts.core.validation import ValidationRules

def get_default_window_size(frequency: str = 'daily') -> int:
    """
    Get default DTW window size based on return frequency.

    Args:
        frequency: 'daily', 'weekly', or 'monthly'

    Returns:
        Recommended window size for DTW calculation
    """
    window_sizes = {
        'daily': 20,    # Capture up to 1 month of trading lag
        'weekly': 8,    # Capture up to 2 months of trading lag
        'monthly': 3    # Capture up to 1 quarter of trading lag
    }
    return window_sizes.get(frequency, 20)

class DTWCorrelation:
    """Handles DTW-based correlation calculation."""

    def __init__(self, frequency: str = 'daily', window_size: Optional[int] = None):
        """
        Initialize DTW correlation calculator.

        Args:
            frequency: Data frequency ('daily', 'weekly', 'monthly')
            window_size: Optional override for default window size
        """
        self.frequency = frequency
        self.window_size = window_size or get_default_window_size(frequency)
        self.similarity_calculator = DTWSimilarity(self.window_size)
        self._setup_aligner()

    def _setup_aligner(self) -> None:
        """Setup data aligner for DTW calculations."""
        self.aligner = DataAligner(
            AlignmentConfig(
                method=AlignmentMethod.PAIRWISE_OVERLAPPING,
                frequency=self.frequency
            )
        )

    def calculate_correlation_matrix(
        self,
        data: pd.DataFrame
    ) -> Tuple[CorrelationCovarianceMatrix, Dict[str, float]]:
        """
        Calculate DTW-based correlation matrix.

        Args:
            data: DataFrame with time series data (datetime index, series as columns)

        Returns:
            Tuple of:
            - Correlation matrix as CorrelationCovarianceMatrix
            - Calculation statistics
        """
        try:
            # Align and clean data
            complete_data = data.dropna(axis=1, how='any')

            # Check sample size - warn but don't prevent
            ValidationRules.check_observations(
                len(complete_data),
                'correlation',
                'dtw'
            )

            aligned_data = self.aligner.align_series(complete_data)

            # Calculate similarity matrices
            similarities, inverse_similarities = self.similarity_calculator.calculate_similarity_matrices(aligned_data)

            # Create similarity matrix object
            sim_matrix = SimilarityMatrix(
                data=similarities,
                similarity_type='dtw',
                inverse_similarities=inverse_similarities
            )

            # Convert to correlation matrix
            corr_matrix = sim_matrix.to_correlation()

            # Calculate statistics
            stats = {
                "n_series": len(complete_data.columns),
                "series_removed": len(data.columns) - len(complete_data.columns),
                "positive_correlations": np.sum(corr_matrix.data > 0) / 2,
                "negative_correlations": np.sum(corr_matrix.data < 0) / 2,
                "mean_correlation": np.mean(corr_matrix.data[~np.eye(len(complete_data.columns), dtype=bool)]),
                "min_correlation": np.min(corr_matrix.data[~np.eye(len(complete_data.columns), dtype=bool)]),
                "max_correlation": np.max(corr_matrix.data[~np.eye(len(complete_data.columns), dtype=bool)])
            }

            return corr_matrix, stats

        except Exception as e:
            raise ProcessingError(f"Failed to calculate DTW correlation matrix: {str(e)}")

```

## Usage Example

```python
# Initialize DTW correlation calculator with default alignment
dtw_corr = DTWCorrelation(frequency='daily')

# Input: DataFrame of returns with datetime index and assets as columns
# e.g., data.index is DatetimeIndex, data.columns are asset identifiers

# Calculate correlation matrix using default alignment
corr_matrix, stats = dtw_corr.calculate_correlation_matrix(data)

# Or specify custom alignment method with fallback behavior
dtw_corr_custom = DTWCorrelation(
    frequency='daily',
    alignment_config=AlignmentConfig(
        method=AlignmentMethod.SYNCHRONIZED_AVERAGE,  # Primary method
        fallback_method=AlignmentMethod.PAIRWISE_OVERLAPPING,  # Used if constraints not met
        frequency='daily',
        gap_threshold=5,  # Maximum gap length for SYNCHRONIZED_AVERAGE
        min_overlap=0.8   # Minimum overlap ratio required
    )
)

# If gap_threshold or min_overlap conditions aren't met:
# - For affected pairs, falls back to PAIRWISE_OVERLAPPING
# - Logs warning about method change
# - Includes method used in statistics

# Compare alignment effects
print("Default alignment stats:")
print(f"Series removed: {stats['series_removed']}")
print(f"Mean correlation: {stats['mean_correlation']:.3f}")

print("\nCustom alignment stats:")
print(f"Series removed: {stats_custom['series_removed']}")
print(f"Mean correlation: {stats_custom['mean_correlation']:.3f}")

# Convert to covariance if needed
volatilities = np.array([0.1, 0.15, 0.2])  # Example volatilities
cov_matrix = corr_matrix.to_covariance(volatilities)
```

### Available Alignment Methods
```python
from fts.data.alignment import AlignmentMethod

# Available methods:
# - PAIRWISE_OVERLAPPING (default): Uses only overlapping periods for each pair
# - SYNCHRONIZED_AVERAGE: Distributes returns across gaps
# - ALL_INCLUDING_GAPS: Includes all periods, allowing NaN values
# - PAIRWISE_AVERAGE: Distributes returns across gaps per pair

# Fallback behavior:
# Default fallback chain:
# 1. SYNCHRONIZED_AVERAGE -> PAIRWISE_AVERAGE -> PAIRWISE_OVERLAPPING
# 2. PAIRWISE_AVERAGE -> PAIRWISE_OVERLAPPING
# 3. ALL_INCLUDING_GAPS -> PAIRWISE_OVERLAPPING
# 4. PAIRWISE_OVERLAPPING (no fallback needed - most conservative method)
#
# Custom fallback can be specified:
alignment_config = AlignmentConfig(
    method=AlignmentMethod.SYNCHRONIZED_AVERAGE,
    fallback_method=AlignmentMethod.ALL_INCLUDING_GAPS,  # Override default fallback
    gap_threshold=5,
    min_overlap=0.8
)
#
# If no fallback specified:
# - Uses default fallback chain
# - Logs warning when falling back
# - Fallback occurs per pair, allowing mixed methods in final matrix
# - Statistics track which method was used for each pair

```

## Implementation Notes

### Key Features
1. Uses fastdtw through DTWSimilarity
2. Proper handling of negative correlations
3. Default window sizes based on data frequency
4. Comprehensive statistics tracking
+ 5. Intelligent alignment method selection
+ 6. Flexible user override options

### Validation Rules
1. Input data must not contain gaps
2. Frequency must be valid
3. Window size must be positive
+ 4. Alignment method must be compatible with data characteristics

### Error Handling
- ValidationError for invalid inputs
- ProcessingError for calculation failures
- Clear error messages
+ - Warnings for suboptimal method selection

+ ### References
+ - DTW_to_CorrelationMatrix.md: Core methodology and alignment strategies
+ - For implementation details of matrix validation and adjustment methods, see sklearn.covariance documentation
