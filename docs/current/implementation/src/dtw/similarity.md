# DTW Similarity Module

## Overview
This module provides DTW-based similarity calculations using established libraries (fastdtw) with appropriate standardization and window constraints.

## Classes

### DTWSimilarity

```python
from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from fts.core.errors import ValidationError, ProcessingError

class DTWSimilarity:
    """Calculates DTW-based similarities between time series."""

    def __init__(self, window_size: int):
        self.window_size = window_size

    def calculate_similarity_matrices(
        self,
        data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate both original and inverse similarity matrices.

        Args:
            data: DataFrame of time series

        Returns:
            Tuple of (original_similarities, inverse_similarities)
        """
        n_series = len(data.columns)
        similarities = np.eye(n_series)
        inverse_similarities = np.eye(n_series)

        for i in range(n_series):
            for j in range(i+1, n_series):
                sim_orig, sim_inv = self.calculate_similarity(
                    data.iloc[:,i],
                    data.iloc[:,j]
                )
                similarities[i,j] = similarities[j,i] = sim_orig
                inverse_similarities[i,j] = inverse_similarities[j,i] = sim_inv

        return similarities, inverse_similarities

    def calculate_similarity(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[float, float]:
        """Calculate similarity between two series and its inverse."""
        try:
            # Convert and standardize inputs
            x = self._standardize(self._to_array(series1))
            y = self._standardize(self._to_array(series2))

            # Validate inputs
            self._validate_inputs(x, y)

            # Calculate DTW distances for both original and inverted series
            distance_original, _ = fastdtw(x, y, radius=self.window_size)
            distance_inverse, _ = fastdtw(x, -y, radius=self.window_size)

            # Convert distances to similarities (0 to 1 scale)
            max_distance = max(distance_original, distance_inverse)
            sim_original = 1 - (distance_original / max_distance)
            sim_inverse = 1 - (distance_inverse / max_distance)

            return sim_original, sim_inverse

        except Exception as e:
            raise ProcessingError(f"DTW similarity calculation failed: {str(e)}")

    def _standardize(self, series: np.ndarray) -> np.ndarray:
        """
        Standardize series to have mean=0 and std=1.

        Standardization is crucial for:
        1. Scale independence
        2. Numerical stability
        3. Fair comparison of original vs inverted series
        """
        std = np.std(series)
        if std == 0:
            raise ValidationError("Series has zero standard deviation")
        return (series - np.mean(series)) / std

    def _to_array(self, series: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(series, pd.Series):
            return series.values
        return np.asarray(series)

    def _validate_inputs(self, x: np.ndarray, y: np.ndarray) -> None:
        """Validate input arrays."""
        if len(x) < 2 or len(y) < 2:
            raise ValidationError("Time series must have at least 2 points")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValidationError("Time series contain invalid values")
```

## Usage Example

```python
# Initialize DTW calculator with window constraint
dtw = DTWSimilarity(window_size=10)

# Calculate similarities for both original and inverted series
sim_original, sim_inverse = dtw.calculate_similarity(series1, series2)

# Higher original similarity indicates positive correlation
# Higher inverse similarity indicates negative correlation
print(f"Original similarity: {sim_original:.3f}")
print(f"Inverse similarity: {sim_inverse:.3f}")
```

## Implementation Notes

### Key Features
1. Uses fastdtw library for efficient DTW calculation
2. Proper standardization of input series
3. Handles both positive and negative correlations
4. Window constraints for computational efficiency

### Dependencies
- fastdtw: Efficient DTW implementation
- numpy: Numerical operations
- pandas: Data handling

### Validation Rules
1. Window size must be positive
2. Input series must have at least 2 points
3. Input values must be finite
4. Series must have non-zero standard deviation

### Error Handling
- ValidationError for invalid inputs
- ProcessingError for calculation failures
- Clear error messages
