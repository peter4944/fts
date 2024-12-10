# DTW Similarity Module

## 1. Overview
This module implements core Dynamic Time Warping (DTW) calculations:
- DTW distance computation
- Series normalization
- Window size optimization
- Distance matrix calculations

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- fastdtw: Efficient DTW implementation
- statistics.base: Basic statistics and normalization

### Related Modules
- dtw/correlation.py: Converts similarities to correlations
- dtw/matrix.py: Builds correlation matrices
- statistics/base.py: Provides normalize_returns function

## 2. Methodology References

### Background Documents
- [DTW_to_CorrelationMatrix.md](../../../references/methodologies/DTW_to_CorrelationMatrix.md)
  * Section 2.1: DTW algorithm
  * Section 2.2: Window size selection
  * Section 2.3: Normalization requirements
  * Section 2.4: Distance calculations

### Mathematical Foundations
#### DTW Algorithm
```python
# DTW Distance
D(i,j) = d(xi, yj) + min(
    D(i-1, j),   # insertion
    D(i, j-1),   # deletion
    D(i-1, j-1)  # match
)

# Window Constraint
|i - j| ≤ window_size

# Normalization
z = (x - μ) / σ
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def calculate_dtw_distance(series1: pd.Series,
                         series2: pd.Series,
                         window_size: int) -> float:
    """
    Calculate DTW distance between two series.

    Args:
        series1: First time series
        series2: Second time series
        window_size: Sakoe-Chiba band width

    Returns:
        DTW distance value

    Notes:
        - Uses normalized series
        - Applies window constraint
        - Returns symmetric distance
    """
    _validate_series(series1, series2)

    # Normalize series
    norm1 = normalize_returns(series1)
    norm2 = normalize_returns(series2)

    # Calculate DTW distance
    distance, _ = fastdtw.fastdtw(
        norm1.values,
        norm2.values,
        radius=window_size
    )
    return distance

def get_window_size(frequency: str) -> int:
    """
    Get optimal window size for given frequency.

    Args:
        frequency: Data frequency ('D', 'W', 'M', etc.)

    Returns:
        Window size in periods

    Notes:
        - Based on empirical studies
        - Considers data frequency
        - Returns conservative estimate
    """
    window_sizes = {
        'D': 21,   # Monthly window
        'W': 8,    # Monthly window
        'M': 4,    # Quarterly window
        'Q': 3,    # Semi-annual window
        'A': 2     # Annual window
    }
    return window_sizes.get(frequency, 21)  # Default to monthly
```

### 3.2 Performance Considerations
- Use fastdtw for efficient computation
- Pre-normalize series
- Cache window sizes
- Optimize for large matrices

### 3.3 Error Handling
```python
def _validate_series(series1: pd.Series,
                    series2: Optional[pd.Series] = None) -> None:
    """Validate input series for DTW calculation."""
    if len(series1) < 2:
        raise ValidationError("Series must have at least 2 observations")
    if not np.isfinite(series1).all():
        raise ValidationError("Series contains non-finite values")
    if series2 is not None and len(series1) != len(series2):
        raise ValidationError("Series must have same length")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Basic DTW Distance
```python
# Calculate DTW distance
window = get_window_size('M')  # Monthly data
dist = calculate_dtw_distance(series1, series2, window)

# Normalize multiple series
normalized = [normalize_series(s) for s in series_list]
```

### 4.2 Parameter Selection
- window_size: Based on data frequency
- normalization: Always required
- radius: Usually same as window_size

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known distance patterns
   - Window constraints
   - Normalization effects

2. Edge Cases
   - Short series
   - Identical series
   - Inverse series
   - Missing values

3. Performance Tests
   - Large series (>1000 points)
   - Multiple calculations
   - Memory usage

## 6. Implementation Status

### Completed Features
- [x] DTW distance calculation
- [x] Series normalization
- [x] Window size optimization
- [x] Input validation

### Known Limitations
- Limited to univariate series
- No adaptive window sizing
- Memory intensive for large series

### Future Enhancements
- Multivariate DTW
- Adaptive window sizing
- GPU acceleration
- Memory optimization
