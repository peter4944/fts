# Data Alignment Module

## 1. Overview
This module implements methods for aligning financial time series with different trading calendars:
- All-overlapping period alignment
- Synchronized average alignment
- Pairwise overlapping alignment
- Pairwise average alignment

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- data.gaps: Gap detection and handling

### Related Modules
- data/gaps.py: Gap detection
- data/loader.py: Data import/export
- statistics/base.py: Basic calculations

## 2. Methodology References

### Background Documents
- [NonOverlappingData.md](../../../references/methodologies/NonOverlappingData.md)
  * Section 2.1: Method 1 - All-overlapping periods
  * Section 2.2: Method 2 - Synchronized average
  * Section 2.3: Method 3 - Pairwise overlapping
  * Section 2.4: Method 4 - Pairwise average

### Mathematical Foundations
```python
# Method 1: All-overlapping periods
common_dates = set.intersection(*[set(series.index) for series in series_list])

# Method 2: Synchronized average
# For each non-overlapping period [t1, t2]:
avg_return = sum(returns_in_period) / len(returns_in_period)

# Method 3: Pairwise overlapping
# For each pair (i,j):
common_dates_ij = set.intersection(set(series_i.index), set(series_j.index))

# Method 4: Pairwise average
# For each pair (i,j) and non-overlapping period [t1, t2]:
avg_return_i = sum(returns_i_in_period) / len(returns_i_in_period)
avg_return_j = sum(returns_j_in_period) / len(returns_j_in_period)
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def align_series_method1(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Align series using all-overlapping periods (Method 1).

    Args:
        returns: DataFrame of return series

    Returns:
        DataFrame with aligned returns (common dates only)

    Notes:
        - Most conservative approach
        - Maximizes data quality
        - May significantly reduce sample size
    """
    return returns.dropna(how='any')

def align_series_method2(returns: pd.DataFrame,
                        min_periods: int = 1) -> pd.DataFrame:
    """
    Align series using synchronized average (Method 2).

    Args:
        returns: DataFrame of return series
        min_periods: Minimum periods for valid average

    Returns:
        DataFrame with averaged returns

    Notes:
        - Averages returns over non-overlapping periods
        - Maintains sample size
        - May smooth out patterns
    """
    # Identify non-overlapping periods
    gaps = detect_gaps(returns)
    periods = get_non_overlapping_periods(gaps)

    # Calculate averages for each period
    aligned_returns = []
    for start, end in periods:
        period_returns = returns.loc[start:end]
        avg_returns = period_returns.mean(min_periods=min_periods)
        aligned_returns.append(avg_returns)

    return pd.DataFrame(aligned_returns)

def align_series_pairwise(returns: pd.DataFrame,
                         method: str = 'overlap') -> Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]]:
    """
    Align series pairwise using either Method 3 or 4.

    Args:
        returns: DataFrame of return series
        method: 'overlap' for Method 3, 'average' for Method 4

    Returns:
        Dictionary mapping asset pairs to their aligned return series
        Format: {('A','B'): (series_A_aligned_to_B, series_B_aligned_to_A), ...}

    Notes:
        - Method 3: Uses only overlapping periods for each pair
        - Method 4: Uses averaged returns in non-overlapping periods
    """
    assets = returns.columns
    pairwise_returns = {}

    for i, asset_i in enumerate(assets):
        for asset_j in assets[i+1:]:
            pair_data = returns[[asset_i, asset_j]]

            if method == 'overlap':
                # Method 3: Keep only overlapping periods
                aligned_pair = pair_data.dropna()
                aligned_i = aligned_pair[asset_i]
                aligned_j = aligned_pair[asset_j]
            else:
                # Method 4: Average non-overlapping periods
                aligned_i, aligned_j = _average_non_overlapping(
                    pair_data[asset_i],
                    pair_data[asset_j]
                )

            pairwise_returns[(asset_i, asset_j)] = (aligned_i, aligned_j)

    return pairwise_returns

def _average_non_overlapping(series_i: pd.Series,
                           series_j: pd.Series,
                           min_periods: int = 1) -> Tuple[pd.Series, pd.Series]:
    """Average returns over non-overlapping periods for a pair of series."""
    # Identify non-overlapping periods
    gaps_i = detect_gaps(series_i)
    gaps_j = detect_gaps(series_j)
    periods = merge_gap_periods(gaps_i, gaps_j)

    # Calculate averages for each period
    avg_i = []
    avg_j = []
    dates = []

    for start, end in periods:
        period_i = series_i.loc[start:end]
        period_j = series_j.loc[start:end]

        if len(period_i) >= min_periods:
            avg_i.append(period_i.mean())
        if len(period_j) >= min_periods:
            avg_j.append(period_j.mean())
        dates.append(end)

    return (pd.Series(avg_i, index=dates),
            pd.Series(avg_j, index=dates))
```

### 3.2 Performance Considerations
- Cache gap detection results
- Optimize period merging
- Handle large datasets efficiently
- Minimize redundant calculations

### 3.3 Error Handling
```python
def _validate_alignment_inputs(returns: pd.DataFrame) -> None:
    """Validate inputs for alignment functions."""
    if not isinstance(returns, pd.DataFrame):
        raise ValidationError("Returns must be a DataFrame")
    if len(returns.columns) < 2:
        raise ValidationError("Need at least 2 series to align")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Method Selection
```python
# Method 1: Maximum quality, minimum quantity
aligned_m1 = align_series_method1(returns)

# Method 2: Balance quality/quantity
aligned_m2 = align_series_method2(returns)

# Method 3: Pairwise overlap
pairwise_m3 = align_series_pairwise(returns, method='overlap')

# Method 4: Pairwise average
pairwise_m4 = align_series_pairwise(returns, method='average')
```

### 4.2 Method Selection Guidelines
- Method 1: When data quality is critical
- Method 2: When sample size matters
- Method 3: For pairwise analysis
- Method 4: For maximum data utilization

### 4.3 Integration with DTW
```python
def prepare_for_dtw(returns: pd.DataFrame,
                    method: str = 'method1',
                    **kwargs) -> Union[pd.DataFrame, Dict[Tuple[str, str], Tuple[pd.Series, pd.Series]]]:
    """
    Prepare return series for DTW correlation calculation.

    Args:
        returns: DataFrame of return series
        method: Alignment method ('method1', 'method2', 'method3', 'method4')
        **kwargs: Additional arguments for alignment methods

    Returns:
        Aligned data in appropriate format for DTW correlation

    Notes:
        - Methods 1/2 return DataFrame
        - Methods 3/4 return Dict of paired series
    """
    if method == 'method1':
        return align_series_method1(returns)
    elif method == 'method2':
        return align_series_method2(returns, **kwargs)
    elif method == 'method3':
        return align_series_pairwise(returns, method='overlap')
    elif method == 'method4':
        return align_series_pairwise(returns, method='average')
    else:
        raise ValueError(f"Unknown method: {method}")
```

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Known alignment patterns
   - Gap handling
   - Average calculations

2. Edge Cases
   - No overlap
   - Single gap
   - Multiple gaps
   - Missing data

3. Performance Tests
   - Large datasets
   - Many series
   - Complex gap patterns

## 6. Implementation Status

### Completed Features
- [x] Method 1: All-overlapping
- [x] Method 2: Synchronized average
- [x] Method 3: Pairwise overlap
- [x] Method 4: Pairwise average
- [x] Gap detection integration
- [x] Input validation

### Known Limitations
- Memory intensive for large datasets
- No streaming support
- Limited calendar handling

### Future Enhancements
- Calendar-aware alignment
- Streaming processing
- Parallel computation
- Custom period aggregation
