# DTW Module Implementation

## 1. Core DTW Functionality

```python
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from ..core.validation import ValidationError

@dataclass(frozen=True)
class DTWParameters:
    """Configuration for DTW calculations."""
    window_size: Optional[int] = None  # If None, determined by frequency
    frequency: str = 'D'  # 'D' for daily, 'W' for weekly, 'M' for monthly
    distance_metric: str = 'euclidean'
    require_standardized: bool = True

    def __post_init__(self):
        """Set default window size based on frequency if not provided."""
        if self.window_size is None:
            # Default window sizes by frequency
            window_sizes = {
                'D': 20,  # Daily: 20 days (approximately 1 month)
                'W': 8,   # Weekly: 8 weeks (approximately 2 months)
                'M': 6    # Monthly: 6 months (half year)
            }
            object.__setattr__(self, 'window_size', window_sizes.get(self.frequency))

class DTWCorrelation:
    """Dynamic Time Warping correlation calculations."""

    def __init__(self, parameters: DTWParameters):
        self.parameters = parameters
        self._validate_parameters()

    def calculate_correlation(self,
                            series1: pd.Series,
                            series2: pd.Series) -> Tuple[float, float]:
        """
        Calculate DTW correlation between two series.

        Returns:
            Tuple of (original_correlation, inverse_correlation)
        """
        # Validate inputs
        if self.parameters.require_standardized:
            if not (self._is_standardized(series1) and self._is_standardized(series2)):
                raise ValidationError("Input series must be standardized")

        # Calculate DTW distance for original series
        dist_orig = self._calculate_dtw_distance(series1, series2)

        # Calculate DTW distance for inverse relationship
        dist_inv = self._calculate_dtw_distance(series1, -series2)

        # Convert distances to similarities [0,1]
        sim_orig = 1 / (1 + dist_orig)
        sim_inv = 1 / (1 + dist_inv)

        return sim_orig, sim_inv

    def build_correlation_matrix(self,
                               series_collection: TimeSeriesCollection
                               ) -> pd.DataFrame:
        """
        Build correlation matrix for multiple series.

        Returns symmetric correlation matrix with ones on diagonal.
        """
        series_names = list(series_collection.series.keys())
        n_series = len(series_names)

        # Initialize correlation matrix
        corr_matrix = np.ones((n_series, n_series))

        # Calculate correlations for upper triangle
        for i in range(n_series):
            for j in range(i+1, n_series):
                series_i = series_collection.series[series_names[i]].data
                series_j = series_collection.series[series_names[j]].data

                # Get correlation (use max of original and inverse)
                sim_orig, sim_inv = self.calculate_correlation(series_i, series_j)
                corr = max(sim_orig, sim_inv)
                if sim_inv > sim_orig:
                    corr = -corr

                # Fill both upper and lower triangle
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr

        return pd.DataFrame(
            corr_matrix,
            index=series_names,
            columns=series_names
        )

    def _calculate_dtw_distance(self,
                              series1: pd.Series,
                              series2: pd.Series) -> float:
        """Calculate DTW distance between two series."""
        # Ensure series are aligned
        common_idx = series1.index.intersection(series2.index)
        s1 = series1.loc[common_idx].values
        s2 = series2.loc[common_idx].values

        # Calculate DTW with window constraint
        distance, _ = fastdtw(
            s1, s2,
            radius=self.parameters.window_size,
            dist=self._get_distance_func()
        )

        return distance

    def _get_distance_func(self):
        """Get distance function based on metric parameter."""
        if self.parameters.distance_metric == 'euclidean':
            return lambda x, y: np.sqrt((x - y) ** 2)
        elif self.parameters.distance_metric == 'squared':
            return lambda x, y: (x - y) ** 2
        elif self.parameters.distance_metric == 'manhattan':
            return lambda x, y: np.abs(x - y)
        else:
            raise ValueError(f"Unsupported distance metric: {self.parameters.distance_metric}")

    def _is_standardized(self, series: pd.Series, tolerance: float = 1e-10) -> bool:
        """Check if series is standardized (mean ≈ 0, std ≈ 1)."""
        return (abs(series.mean()) < tolerance and
                abs(series.std() - 1) < tolerance)

    def _validate_parameters(self) -> None:
        """Validate DTW parameters."""
        if self.parameters.frequency not in ['D', 'W', 'M']:
            raise ValidationError(f"Unsupported frequency: {self.parameters.frequency}")

        if self.parameters.window_size is not None and self.parameters.window_size <= 0:
            raise ValidationError("Window size must be positive")

        if self.parameters.distance_metric not in ['euclidean', 'squared', 'manhattan']:
            raise ValidationError(f"Unsupported distance metric: {self.parameters.distance_metric}")
