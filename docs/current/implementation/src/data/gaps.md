# Gap Detection and Handling Module

## 1. Overview
This module implements gap detection and analysis for financial time series:
- Gap detection and classification
- Gap pattern analysis
- Gap filling methods
- Gap period merging

### Core Dependencies
- numpy: Numerical computations
- pandas: Time series handling
- statistics.base: Basic calculations

### Related Modules
- data/alignment.py: Uses gap information for alignment
- data/loader.py: Initial data validation
- statistics/base.py: Basic calculations

## 2. Methodology References

### Background Documents
- [NonOverlappingData.md](../../../references/methodologies/NonOverlappingData.md)
  * Section 1.2: Gap identification
  * Section 1.3: Gap classification
  * Section 1.4: Gap handling strategies

### Mathematical Foundations
```python
# Gap Definition
gap = {
    'start': t where data[t] exists and data[t+1] missing,
    'end': t where data[t-1] missing and data[t] exists,
    'length': number of missing periods
}

# Gap Classification
small_gap = gap where length <= max_small_gap
large_gap = gap where length > max_small_gap

# Period Merging
merged_period = [min(start_1, start_2), max(end_1, end_2)]
if gap_between(period_1, period_2) <= max_merge_gap
```

## 3. Implementation Details

### 3.1 Core Functions
```python
def detect_gaps(series: pd.Series,
                freq: Optional[str] = None) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Detect gaps in time series data.

    Args:
        series: Time series data
        freq: Optional frequency (inferred if None)

    Returns:
        List of (gap_start, gap_end) timestamps

    Notes:
        - Gaps are periods with missing data
        - Start is last valid data before gap
        - End is first valid data after gap
    """
    if freq is None:
        freq = pd.infer_freq(series.index)

    # Create full date range
    full_index = pd.date_range(
        start=series.index.min(),
        end=series.index.max(),
        freq=freq
    )

    # Find missing dates
    missing = full_index.difference(series.index)

    # Convert to gap periods
    gaps = []
    if len(missing) > 0:
        gap_groups = np.split(missing, np.where(np.diff(missing) > pd.Timedelta(freq))[0] + 1)

        for group in gap_groups:
            if len(group) > 0:
                gap_start = series.index[series.index < group[0]].max()
                gap_end = series.index[series.index > group[-1]].min()
                gaps.append((gap_start, gap_end))

    return gaps

def analyze_gap_patterns(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze gap patterns in time series.

    Args:
        series: Time series data

    Returns:
        Dictionary with gap statistics:
        - total_gaps: Number of gaps
        - avg_gap_length: Average gap length
        - max_gap_length: Maximum gap length
        - gap_distribution: Gap length distribution

    Notes:
        - Provides insights for alignment strategy
        - Helps identify problematic periods
    """
    gaps = detect_gaps(series)

    if not gaps:
        return {
            'total_gaps': 0,
            'avg_gap_length': 0,
            'max_gap_length': 0,
            'gap_distribution': {}
        }

    # Calculate gap lengths
    gap_lengths = [(end - start).days for start, end in gaps]

    # Analyze distribution
    gap_dist = pd.Series(gap_lengths).value_counts().to_dict()

    return {
        'total_gaps': len(gaps),
        'avg_gap_length': np.mean(gap_lengths),
        'max_gap_length': max(gap_lengths),
        'gap_distribution': gap_dist
    }

def merge_gap_periods(gaps1: List[Tuple[pd.Timestamp, pd.Timestamp]],
                     gaps2: List[Tuple[pd.Timestamp, pd.Timestamp]],
                     max_merge_gap: int = 1) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Merge gap periods from two series.

    Args:
        gaps1: Gaps from first series
        gaps2: Gaps from second series
        max_merge_gap: Maximum gap for merging periods

    Returns:
        List of merged gap periods

    Notes:
        - Combines overlapping periods
        - Merges nearby periods within max_merge_gap
        - Maintains chronological order
    """
    # Combine all periods
    all_periods = gaps1 + gaps2
    if not all_periods:
        return []

    # Sort by start date
    all_periods.sort(key=lambda x: x[0])

    # Merge overlapping or nearby periods
    merged = [all_periods[0]]
    for current in all_periods[1:]:
        prev = merged[-1]

        # Check if periods should be merged
        gap_size = (current[0] - prev[1]).days
        if gap_size <= max_merge_gap:
            merged[-1] = (prev[0], max(prev[1], current[1]))
        else:
            merged.append(current)

    return merged

def get_non_overlapping_periods(gaps: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Get non-overlapping periods between gaps.

    Args:
        gaps: List of gap periods

    Returns:
        List of valid data periods

    Notes:
        - Returns periods between gaps
        - Used for period-based alignment
        - Excludes gap periods
    """
    if not gaps:
        return []

    # Sort gaps
    sorted_gaps = sorted(gaps, key=lambda x: x[0])

    # Find periods between gaps
    periods = []
    for i in range(len(sorted_gaps) - 1):
        period_start = sorted_gaps[i][1]
        period_end = sorted_gaps[i + 1][0]
        periods.append((period_start, period_end))

    return periods
```

### 3.2 Performance Considerations
- Cache gap detection results
- Optimize period merging
- Handle large datasets efficiently
- Minimize date range operations

### 3.3 Error Handling
```python
def _validate_gap_inputs(series: pd.Series) -> None:
    """Validate inputs for gap detection."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValidationError("Series must have DatetimeIndex")
    if series.index.duplicated().any():
        raise ValidationError("Index contains duplicates")
```

## 4. Usage Guidelines

### 4.1 Common Use Cases

#### Gap Analysis
```python
# Detect gaps
gaps = detect_gaps(series)

# Analyze patterns
gap_stats = analyze_gap_patterns(series)

# Get valid periods
valid_periods = get_non_overlapping_periods(gaps)
```

### 4.2 Parameter Selection
- max_small_gap: Based on data frequency
- max_merge_gap: Based on analysis needs
- freq: Match data collection frequency

## 5. Testing Requirements

### Coverage Requirements
- All functions must have >95% test coverage
- Edge cases must be explicitly tested
- Performance benchmarks must be maintained

### Critical Test Cases
1. Basic Functionality
   - Simple gaps
   - Multiple gaps
   - Period merging

2. Edge Cases
   - No gaps
   - Single gap
   - Adjacent gaps
   - Overlapping periods

3. Performance Tests
   - Large datasets
   - Many gaps
   - Complex merging

## 6. Implementation Status

### Completed Features
- [x] Gap detection
- [x] Pattern analysis
- [x] Period merging
- [x] Non-overlapping periods

### Known Limitations
- Limited calendar awareness
- No streaming support
- Memory intensive for large datasets

### Future Enhancements
- Calendar-aware gap detection
- Streaming processing
- Advanced gap statistics
- Custom gap classifications
