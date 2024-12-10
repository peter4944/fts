# DTW Correlation Module

## 1. Overview
This module converts DTW distances to correlation-like measures:
- DTW distance to correlation conversion
- Negative correlation handling
- Correlation normalization
- Similarity score adjustments

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- statistics.base: Basic statistical calculations
- dtw.similarity: DTW distance calculations

### Related Modules
- dtw/similarity.py: Core DTW calculations
- dtw/matrix.py: Correlation matrix construction
- statistics/base.py: Statistical utilities

## 2. Methodology References

### Background Documents
- [DTW_to_CorrelationMatrix.md](../../../references/methodologies/DTW_to_CorrelationMatrix.md)
  * Section 3.1: Distance to correlation conversion
  * Section 3.2: Negative correlation detection
  * Section 3.3: Normalization requirements
  * Section 3.4: Correlation properties

### Mathematical Foundations
#### Correlation Conversion
```python
# For normalized series (from similarity module):
# Step 1: Get minimum distance (original or inverse)
min_dist = min(distance, inverse_distance)

# Step 2: Convert to similarity
similarity = 1 - min_dist

# Step 3: Convert to correlation [-1, 1]
correlation = 2 * similarity - 1

# Negative correlation detection
neg_corr = sign * correlation
where sign = -1 if series are inverse, 1 otherwise
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def dtw_to_correlation(similarity: float,
                      inverse_similarity: float) -> float:
    """
    Convert DTW similarity to correlation-like measure.

    Args:
        similarity: DTW similarity score
        inverse_similarity: DTW score for inverse series

    Returns:
        Correlation value in [-1, 1]

    Notes:
        - Input distances from normalized series
        - Converts to similarity in [0,1]
        - Maps to correlation in [-1,1]
        - Handles negative correlations via inverse series
    """
    _validate_similarities(similarity, inverse_similarity)

    # Determine if correlation should be negative
    is_negative = similarity > inverse_similarity

    # Get minimum distance (best match)
    min_dist = min(similarity, inverse_similarity)

    # Convert distance to similarity [0,1]
    sim = 1 - min_dist

    # Map to correlation [-1,1]
    correlation = 2 * sim - 1

    # Apply sign
    return -correlation if is_negative else correlation

def handle_negative_correlation(series1: pd.Series,
                             series2: pd.Series) -> float:
    """
    Detect and handle negative correlations.

    Args:
        series1: First time series
        series2: Second time series

    Returns:
        Sign multiplier (-1 or 1)

    Notes:
        - Compares original and inverted series
        - Uses normalized series
        - Considers window constraints
    """
    _validate_series(series1, series2)

    # Get DTW distances for original and inverted
    window = get_window_size(_infer_frequency(series1))
    dist_orig = calculate_dtw_distance(series1, series2, window)
    dist_inv = calculate_dtw_distance(series1, -series2, window)

    return -1 if dist_inv < dist_orig else 1
```

### 3.2 Performance Considerations
- Cache similarity calculations
- Reuse window sizes
- Optimize inverse calculations
- Handle large matrices efficiently

### 3.3 Error Handling
```python
def _validate_similarities(sim: float,
                         inv_sim: float) -> None:
    """Validate similarity scores."""
    if not (np.isfinite(sim) and np.isfinite(inv_sim)):
        raise ValidationError("Similarities must be finite")
    if sim < 0 or inv_sim < 0:
        raise ValidationError("Similarities must be non-negative")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Basic Correlation Conversion
```python
# Calculate correlations
window = get_window_size('M')
dist = calculate_dtw_distance(series1, series2, window)
dist_inv = calculate_dtw_distance(series1, -series2, window)
corr = dtw_to_correlation(dist, dist_inv)
```

### 4.2 Parameter Selection
- similarity_threshold: Based on data characteristics
- window_size: From similarity module
- normalization: Required for comparison

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known correlation patterns
   - Sign detection
   - Normalization effects

2. Edge Cases
   - Perfect correlation/anti-correlation
   - No correlation
   - Boundary cases

3. Performance Tests
   - Large series
   - Multiple conversions
   - Memory usage

## 6. Implementation Status

### Completed Features
- [x] Distance to correlation conversion
- [x] Negative correlation handling
- [x] Input validation
- [x] Performance optimization

### Known Limitations
- Assumes monotonic relationship
- No partial correlations
- Memory intensive for large matrices

### Future Enhancements
- Advanced correlation measures
- Partial correlation support
- Memory optimization
- Parallel processing
